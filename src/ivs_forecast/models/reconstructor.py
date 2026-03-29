from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ivs_forecast.models.base import assert_cuda_available

M_MIN = math.log(0.6)
M_MAX = math.log(2.0)
TAU_MAX = 730.0 / 365.0
GRID_SIZE = 154


@dataclass(frozen=True)
class ReconstructorTrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    penalty_lambda: float


def load_training_config(path: Path) -> ReconstructorTrainConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    training = payload["training"]
    return ReconstructorTrainConfig(
        batch_size=training["batch_size"],
        epochs=training["epochs"],
        learning_rate=training["learning_rate"],
        penalty_lambda=training["penalty_lambda"],
    )


class ReconstructorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GRID_SIZE + 2, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class _SurfaceNodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        surfaces: np.ndarray,
        date_indices: np.ndarray,
        m: np.ndarray,
        tau: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.surfaces = torch.tensor(surfaces, dtype=torch.float32)
        self.date_indices = torch.tensor(date_indices, dtype=torch.int64)
        self.m = torch.tensor(m, dtype=torch.float32)
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.date_indices.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.date_indices[index],
            self.m[index],
            self.tau[index],
            self.target[index],
        )


@dataclass(frozen=True)
class PenaltyGrid:
    ic34_m: np.ndarray
    ic34_tau: np.ndarray
    ic5_m: np.ndarray
    ic5_tau: np.ndarray


def build_penalty_grid() -> PenaltyGrid:
    cubic_start = -((-2.0 * M_MIN) ** (1.0 / 3.0))
    cubic_end = (2.0 * M_MAX) ** (1.0 / 3.0)
    cubic_axis = np.linspace(cubic_start, cubic_end, 40, dtype=np.float64)
    ic34_m_axis = cubic_axis**3
    tau_axis = np.exp(
        np.linspace(math.log(1.0 / 365.0), math.log(TAU_MAX + 1.0), 40, dtype=np.float64)
    )
    ic34_m, ic34_tau = np.meshgrid(ic34_m_axis, tau_axis, indexing="xy")
    ic5_m, ic5_tau = np.meshgrid(
        np.array([6.0 * M_MIN, 4.0 * M_MIN, 4.0 * M_MAX, 6.0 * M_MAX], dtype=np.float64),
        tau_axis,
        indexing="xy",
    )
    return PenaltyGrid(
        ic34_m=ic34_m.reshape(-1),
        ic34_tau=ic34_tau.reshape(-1),
        ic5_m=ic5_m.reshape(-1),
        ic5_tau=ic5_tau.reshape(-1),
    )


def _query_with_derivatives(
    network: ReconstructorNetwork,
    sampled_surface: torch.Tensor,
    m: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.cat([sampled_surface, m.unsqueeze(1), tau.unsqueeze(1)], dim=1)
    sigma = network(query).squeeze(1)
    grad_outputs = torch.ones_like(sigma)
    sigma_m, sigma_tau = torch.autograd.grad(
        sigma,
        (m, tau),
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )
    sigma_mm = torch.autograd.grad(
        sigma_m,
        m,
        grad_outputs=torch.ones_like(sigma_m),
        create_graph=True,
        retain_graph=True,
    )[0]
    return sigma, sigma_m, sigma_tau, sigma_mm


def _repeat_surfaces(
    surfaces: torch.Tensor, m: np.ndarray, tau: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = surfaces.shape[0]
    m_tensor = torch.tensor(
        np.tile(m, count), dtype=torch.float32, device=device, requires_grad=True
    )
    tau_tensor = torch.tensor(
        np.tile(tau, count), dtype=torch.float32, device=device, requires_grad=True
    )
    repeated_surfaces = surfaces.repeat_interleave(len(m), dim=0)
    return repeated_surfaces, m_tensor, tau_tensor


def _penalty_terms(
    network: ReconstructorNetwork,
    unique_surfaces: torch.Tensor,
    penalty_grid: PenaltyGrid,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated_surfaces, m34, tau34 = _repeat_surfaces(
        unique_surfaces, penalty_grid.ic34_m, penalty_grid.ic34_tau, device
    )
    sigma34, sigma_m34, sigma_tau34, sigma_mm34 = _query_with_derivatives(
        network, repeated_surfaces, m34, tau34
    )
    phi_cal = sigma34 + 2.0 * tau34 * sigma_tau34
    phi_but = (
        1.0
        - ((m34 * sigma_m34 / sigma34) ** 2)
        - (((sigma34 * tau34 * sigma_m34) ** 2) / 4.0)
        + (tau34 * sigma34 * sigma_mm34)
    )
    lc3 = torch.relu(-phi_cal).mean()
    lc4 = torch.relu(-phi_but).mean()
    repeated_surfaces_ic5, m5, tau5 = _repeat_surfaces(
        unique_surfaces, penalty_grid.ic5_m, penalty_grid.ic5_tau, device
    )
    sigma5, sigma_m5, _, sigma_mm5 = _query_with_derivatives(
        network, repeated_surfaces_ic5, m5, tau5
    )
    lc5 = torch.abs(sigma5 * sigma_mm5 + sigma_m5**2).mean()
    return lc3, lc4, lc5


@dataclass
class FittedReconstructor:
    network: ReconstructorNetwork
    device: torch.device
    penalty_grid: PenaltyGrid

    def predict(
        self, sampled_surface: np.ndarray, m: np.ndarray, tau: np.ndarray, batch_size: int = 4096
    ) -> np.ndarray:
        self.network.eval()
        sampled = torch.tensor(sampled_surface.astype(np.float32), device=self.device).unsqueeze(0)
        outputs: list[np.ndarray] = []
        for start in range(0, len(m), batch_size):
            stop = min(start + batch_size, len(m))
            m_batch = torch.tensor(m[start:stop], dtype=torch.float32, device=self.device)
            tau_batch = torch.tensor(tau[start:stop], dtype=torch.float32, device=self.device)
            surfaces = sampled.repeat(stop - start, 1)
            with torch.no_grad():
                preds = self.network(
                    torch.cat([surfaces, m_batch.unsqueeze(1), tau_batch.unsqueeze(1)], dim=1)
                )
            outputs.append(preds.squeeze(1).detach().cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)

    def arbitrage_diagnostics(self, sampled_surface: np.ndarray) -> dict[str, float]:
        self.network.eval()
        surface = torch.tensor(sampled_surface.astype(np.float32), device=self.device).unsqueeze(0)
        lc3, lc4, _ = _penalty_terms(self.network, surface, self.penalty_grid, self.device)
        repeated_surfaces, m34, tau34 = _repeat_surfaces(
            surface, self.penalty_grid.ic34_m, self.penalty_grid.ic34_tau, self.device
        )
        sigma34, sigma_m34, sigma_tau34, sigma_mm34 = _query_with_derivatives(
            self.network, repeated_surfaces, m34, tau34
        )
        phi_cal = sigma34 + 2.0 * tau34 * sigma_tau34
        phi_but = (
            1.0
            - ((m34 * sigma_m34 / sigma34) ** 2)
            - (((sigma34 * tau34 * sigma_m34) ** 2) / 4.0)
            + (tau34 * sigma34 * sigma_mm34)
        )
        negative_cal = torch.minimum(phi_cal, torch.zeros_like(phi_cal))
        negative_but = torch.minimum(phi_but, torch.zeros_like(phi_but))
        violating_fraction = ((phi_cal < 0.0) | (phi_but < 0.0)).float().mean()
        return {
            "mean_negative_calendar_mass": float(negative_cal.mean().detach().cpu().item()),
            "mean_negative_butterfly_mass": float(negative_but.mean().detach().cpu().item()),
            "violating_fraction": float(violating_fraction.detach().cpu().item()),
            "calendar_penalty": float(lc3.detach().cpu().item()),
            "butterfly_penalty": float(lc4.detach().cpu().item()),
        }


def train_reconstructor(
    sampled_surface_wide: pl.DataFrame,
    surface_nodes: pl.DataFrame,
    available_dates: list[object],
    config_path: Path,
    seed: int,
) -> FittedReconstructor:
    assert_cuda_available("reconstructor")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    config = load_training_config(config_path)
    sampled_train = sampled_surface_wide.filter(pl.col("quote_date").is_in(available_dates)).sort(
        "quote_date"
    )
    nodes_train = surface_nodes.filter(pl.col("quote_date").is_in(available_dates)).sort(
        "quote_date"
    )
    if sampled_train.is_empty() or nodes_train.is_empty():
        raise ValueError("Reconstructor training received no eligible training dates.")
    iv_columns = [f"iv_g{index:03d}" for index in range(GRID_SIZE)]
    surface_matrix = sampled_train.select(iv_columns).to_numpy().astype(np.float32)
    date_to_index = {value: idx for idx, value in enumerate(sampled_train["quote_date"].to_list())}
    dataset = _SurfaceNodeDataset(
        surfaces=surface_matrix,
        date_indices=np.array(
            [date_to_index[value] for value in nodes_train["quote_date"].to_list()], dtype=np.int64
        ),
        m=nodes_train["m"].to_numpy().astype(np.float32),
        tau=nodes_train["tau"].to_numpy().astype(np.float32),
        target=nodes_train["node_iv"].to_numpy().astype(np.float32),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    network = ReconstructorNetwork().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    penalty_grid = build_penalty_grid()
    for _epoch in range(config.epochs):
        network.train()
        for date_indices, m, tau, target in loader:
            date_indices = date_indices.to(device)
            m = m.to(device).requires_grad_(True)
            tau = tau.to(device).requires_grad_(True)
            target = target.to(device)
            surface_batch = dataset.surfaces[date_indices.cpu()].to(device)
            optimizer.zero_grad(set_to_none=True)
            sigma, _, _, _ = _query_with_derivatives(network, surface_batch, m, tau)
            mse = loss_fn(sigma, target)
            unique_indices = torch.unique(date_indices).detach().cpu().numpy()
            unique_surfaces = dataset.surfaces[unique_indices].to(device)
            lc3, lc4, lc5 = _penalty_terms(network, unique_surfaces, penalty_grid, device)
            loss = mse + config.penalty_lambda * (lc3 + lc4 + lc5)
            loss.backward()
            optimizer.step()
    return FittedReconstructor(network=network, device=device, penalty_grid=penalty_grid)
