from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, Dataset

from ivs_forecast.data.partitioned import DatePartitionIndex
from ivs_forecast.data.ssvi import raw_to_constrained_params, ssvi_implied_vol
from ivs_forecast.models.base import (
    DailyStateStore,
    ModelArtifact,
    NormalizationStats,
    SsviStateModel,
    assert_cuda_available,
    fit_normalization,
    history_feature_columns,
)


@dataclass(frozen=True)
class SsviTcnParams:
    history_days: int
    hidden_width: int
    dropout: float
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 150
    patience: int = 15
    gradient_clip: float = 1.0
    batch_size: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_search_space() -> list[SsviTcnParams]:
    return [
        SsviTcnParams(history_days=history_days, hidden_width=hidden_width, dropout=dropout)
        for history_days in (10, 22)
        for hidden_width in (32, 64)
        for dropout in (0.0, 0.1)
    ]


def masked_vega_huber_loss(
    predicted_iv: torch.Tensor,
    target_iv: torch.Tensor,
    target_vega: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1e-2,
) -> torch.Tensor:
    elementwise = torch_functional.huber_loss(
        predicted_iv,
        target_iv,
        delta=delta,
        reduction="none",
    )
    weights = target_vega * mask
    return torch.sum(weights * elementwise) / weights.sum().clamp_min(1e-12)


class _SsviSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: pl.DataFrame,
        state_store: DailyStateStore,
        surface_nodes_store: DatePartitionIndex,
        history_days: int,
        normalization: NormalizationStats,
    ) -> None:
        self.rows = rows.sort("quote_date").iter_rows(named=True)
        self._rows = list(self.rows)
        self.state_store = state_store
        self.surface_nodes_store = surface_nodes_store
        self.history_days = history_days
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        history = self.state_store.history_tensor(int(row["history_end_index"]), self.history_days)
        history = self.normalization.apply(history.astype(np.float64)).astype(np.float32)
        nodes = self.surface_nodes_store.load_date(row["target_date"])
        return {
            "history": torch.from_numpy(history),
            "target_m": torch.from_numpy(nodes["m"].to_numpy().astype(np.float32)),
            "target_tau": torch.from_numpy(nodes["tau"].to_numpy().astype(np.float32)),
            "target_iv": torch.from_numpy(nodes["node_iv"].to_numpy().astype(np.float32)),
            "target_vega": torch.from_numpy(nodes["node_vega"].to_numpy().astype(np.float32)),
        }


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    histories = torch.stack([item["history"] for item in batch], dim=0)
    max_nodes = max(int(item["target_m"].shape[0]) for item in batch)
    batch_size = len(batch)
    target_m = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    target_tau = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    target_iv = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    target_vega = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    for index, item in enumerate(batch):
        length = int(item["target_m"].shape[0])
        target_m[index, :length] = item["target_m"]
        target_tau[index, :length] = item["target_tau"]
        target_iv[index, :length] = item["target_iv"]
        target_vega[index, :length] = item["target_vega"]
        mask[index, :length] = 1.0
    return {
        "history": histories,
        "target_m": target_m,
        "target_tau": target_tau,
        "target_iv": target_iv,
        "target_vega": target_vega,
        "mask": mask,
    }


def _causal_conv1d(inputs: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
    left_padding = (layer.kernel_size[0] - 1) * layer.dilation[0]
    padded = torch_functional.pad(inputs, (left_padding, 0))
    return layer(padded)


class _ResidualCausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.norm1(inputs).transpose(1, 2)
        outputs = _causal_conv1d(outputs, self.conv1).transpose(1, 2)
        outputs = self.dropout(torch_functional.gelu(outputs))
        outputs = self.norm2(outputs).transpose(1, 2)
        outputs = _causal_conv1d(outputs, self.conv2).transpose(1, 2)
        outputs = self.dropout(torch_functional.gelu(outputs))
        return residual + outputs


class SsviTcnNetwork(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, width)
        self.projection_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [_ResidualCausalBlock(width, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 14),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        outputs = self.projection_norm(self.projection(history))
        for block in self.blocks:
            outputs = block(outputs)
        return self.head(outputs[:, -1, :])


def _split_train_holdout(rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if rows.height < 6:
        raise ValueError("ssvi_tcn_direct requires at least six training rows.")
    holdout_size = max(1, rows.height // 10)
    holdout = rows.tail(holdout_size)
    train = rows.head(rows.height - holdout_size)
    if train.is_empty():
        raise ValueError("ssvi_tcn_direct internal holdout split left no training rows.")
    return train, holdout


class SsviTcnDirectModel(SsviStateModel):
    model_name = "ssvi_tcn_direct"

    def __init__(self, params: SsviTcnParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.network: SsviTcnNetwork | None = None
        self.normalization: NormalizationStats | None = None
        self.best_state_dict: dict[str, Any] | None = None

    def _seed_everything(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(
        self,
        train_rows: pl.DataFrame,
        state_store: DailyStateStore,
        surface_nodes_store: DatePartitionIndex | None = None,
    ) -> None:
        if surface_nodes_store is None:
            raise ValueError("ssvi_tcn_direct requires surface_nodes_store for node-loss training.")
        assert_cuda_available(self.model_name)
        self._seed_everything()
        fit_rows, holdout_rows = _split_train_holdout(train_rows.sort("quote_date"))
        self.normalization = fit_normalization(fit_rows, state_store, self.params.history_days)
        device = torch.device("cuda")
        self.network = SsviTcnNetwork(
            input_dim=len(history_feature_columns()),
            width=self.params.hidden_width,
            dropout=self.params.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        train_loader = DataLoader(
            _SsviSequenceDataset(
                fit_rows,
                state_store,
                surface_nodes_store,
                self.params.history_days,
                self.normalization,
            ),
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=collate_sequence_batch,
        )
        holdout_loader = DataLoader(
            _SsviSequenceDataset(
                holdout_rows,
                state_store,
                surface_nodes_store,
                self.params.history_days,
                self.normalization,
            ),
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=collate_sequence_batch,
        )
        best_loss = float("inf")
        epochs_without_improvement = 0
        for _epoch in range(self.params.max_epochs):
            self.network.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                history = batch["history"].to(device)
                predicted_state = self.network(history)
                constrained = raw_to_constrained_params(predicted_state)
                predicted_iv = ssvi_implied_vol(
                    batch["target_m"].to(device),
                    batch["target_tau"].to(device),
                    constrained,
                )
                loss = masked_vega_huber_loss(
                    predicted_iv,
                    batch["target_iv"].to(device),
                    batch["target_vega"].to(device),
                    batch["mask"].to(device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.params.gradient_clip)
                optimizer.step()
            validation_loss = self._evaluate_loader(holdout_loader, device)
            if validation_loss + 1e-8 < best_loss:
                best_loss = validation_loss
                epochs_without_improvement = 0
                self.best_state_dict = deepcopy(self.network.state_dict())
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.params.patience:
                break
        if self.best_state_dict is None or self.network is None:
            raise RuntimeError("ssvi_tcn_direct failed to produce a valid checkpoint.")
        self.network.load_state_dict(self.best_state_dict)

    def _evaluate_loader(self, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> float:
        if self.network is None:
            raise RuntimeError("ssvi_tcn_direct cannot evaluate before fitting.")
        self.network.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                predicted_state = self.network(batch["history"].to(device))
                constrained = raw_to_constrained_params(predicted_state)
                predicted_iv = ssvi_implied_vol(
                    batch["target_m"].to(device),
                    batch["target_tau"].to(device),
                    constrained,
                )
                loss = masked_vega_huber_loss(
                    predicted_iv,
                    batch["target_iv"].to(device),
                    batch["target_vega"].to(device),
                    batch["mask"].to(device),
                )
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    def predict(
        self,
        feature_rows: pl.DataFrame,
        state_store: DailyStateStore,
    ) -> np.ndarray:
        if self.network is None or self.normalization is None:
            raise RuntimeError("ssvi_tcn_direct must be fit before prediction.")
        assert_cuda_available(self.model_name)
        device = torch.device("cuda")
        self.network.eval()
        histories = []
        for row in feature_rows.iter_rows(named=True):
            history = state_store.history_tensor(int(row["history_end_index"]), self.params.history_days)
            history = self.normalization.apply(history.astype(np.float64)).astype(np.float32)
            histories.append(history)
        history_tensor = torch.from_numpy(np.stack(histories, axis=0)).to(device)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, history_tensor.shape[0], self.params.batch_size):
                batch = history_tensor[start : start + self.params.batch_size]
                outputs.append(self.network(batch).cpu().numpy().astype(np.float64))
        return np.vstack(outputs)

    def save_checkpoint(self, path: str | Any) -> None:
        if self.network is None or self.normalization is None or self.best_state_dict is None:
            raise RuntimeError("ssvi_tcn_direct cannot save a checkpoint before fitting.")
        torch.save(
            {
                "model_name": self.model_name,
                "params": self.params.to_dict(),
                "normalization": self.normalization.to_dict(),
                "feature_columns": history_feature_columns(),
                "state_dict": self.best_state_dict,
            },
            path,
        )

    def artifact(self) -> ModelArtifact:
        params = self.params.to_dict()
        if self.normalization is not None:
            params["normalization"] = self.normalization.to_dict()
        return ModelArtifact(model_name=self.model_name, params=params)
