from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ivs_forecast.models.base import (
    ModelArtifact,
    SampledSurfaceModel,
    assert_cuda_available,
    sequence_matrix,
    target_matrix,
)


@dataclass(frozen=True)
class LstmDirectParams:
    hidden_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    layers: int = 2
    max_epochs: int = 100
    early_stopping_patience: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "layers": self.layers,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }


def load_search_space(config_path: Path) -> list[LstmDirectParams]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid = payload["search_space"]
    fixed = payload["fixed"]
    result = []
    for values in product(
        grid["hidden_size"],
        grid["dropout"],
        grid["learning_rate"],
        grid["batch_size"],
    ):
        result.append(
            LstmDirectParams(
                hidden_size=values[0],
                dropout=values[1],
                learning_rate=values[2],
                batch_size=values[3],
                layers=fixed["layers"],
                max_epochs=fixed["max_epochs"],
                early_stopping_patience=fixed["early_stopping_patience"],
            )
        )
    return result


class _LstmForecaster(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, layers: int, dropout: float, output_size: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(batch)
        return self.output(hidden[-1])


class LstmDirectModel(SampledSurfaceModel):
    model_name = "lstm_direct"

    def __init__(self, params: LstmDirectParams, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.network: _LstmForecaster | None = None

    def fit(self, train_frame: pl.DataFrame) -> None:
        assert_cuda_available(self.model_name)
        device = torch.device("cuda")
        torch.manual_seed(self.seed)
        sequence = sequence_matrix(train_frame)
        target = target_matrix(train_frame).astype(np.float32)
        val_size = max(1, int(0.1 * len(sequence)))
        if len(sequence) <= val_size:
            raise ValueError("lstm_direct needs more than one batch worth of training rows.")
        train_seq, val_seq = sequence[:-val_size], sequence[-val_size:]
        train_target, val_target = target[:-val_size], target[-val_size:]
        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_seq), torch.tensor(train_target)),
            batch_size=self.params.batch_size,
            shuffle=False,
        )
        val_x = torch.tensor(val_seq, device=device)
        val_y = torch.tensor(val_target, device=device)
        network = _LstmForecaster(
            input_size=sequence.shape[2],
            hidden_size=self.params.hidden_size,
            layers=self.params.layers,
            dropout=self.params.dropout,
            output_size=target.shape[1],
        ).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.params.learning_rate)
        loss_fn = nn.MSELoss()
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        patience = 0
        for _epoch in range(self.params.max_epochs):
            network.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(network(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(network(val_x), val_y).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu() for key, value in network.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= self.params.early_stopping_patience:
                    break
        if best_state is None:
            raise RuntimeError("lstm_direct failed to produce a valid checkpoint.")
        network.load_state_dict(best_state)
        self.network = network

    def predict(self, feature_frame: pl.DataFrame) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("lstm_direct must be fit before prediction.")
        device = torch.device("cuda")
        batch = torch.tensor(sequence_matrix(feature_frame), device=device)
        self.network.eval()
        with torch.no_grad():
            preds = self.network(batch).detach().cpu().numpy()
        return preds.astype(np.float64)

    def artifact(self) -> ModelArtifact:
        return ModelArtifact(model_name=self.model_name, params=self.params.to_dict())
