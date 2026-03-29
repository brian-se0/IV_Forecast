from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import torch

from ivs_forecast.config import AppConfig
from ivs_forecast.evaluation.dm import diebold_mariano
from ivs_forecast.features.dataset import build_features_targets
from ivs_forecast.models.reconstructor import (
    ReconstructorNetwork,
    _penalty_terms,
    build_penalty_grid,
)
from ivs_forecast.pipeline.splits import build_split_manifest


def test_build_features_targets_and_split_manifest() -> None:
    rows = []
    for index in range(30):
        row = {
            "quote_date": date(2020, 1, 2) + timedelta(days=index),
            "surface_node_count": 50,
            "valid_expiry_count": 5,
            "active_underlying_price_1545": 100.0 + index,
            "total_trade_volume": 1000 + index,
            "total_open_interest": 2000 + index,
            "median_rel_spread_1545": 0.01,
        }
        for grid_index in range(154):
            value = 0.2 + 0.0001 * grid_index + 0.0005 * index
            row[f"iv_g{grid_index:03d}"] = value
            row[f"logiv_g{grid_index:03d}"] = float(np.log(value))
        rows.append(row)
    sampled = pl.DataFrame(rows)
    features = build_features_targets(sampled)
    assert features.height == 7
    config = AppConfig.model_validate(
        {
            "paths": {"raw_data_root": "D:/Options Data", "artifact_root": "./artifacts"},
            "study": {
                "underlying_symbol": "^SPX",
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "forecast_horizon_days": 1,
            },
            "split": {
                "validation_size": 2,
                "test_size": 2,
                "min_train_size": 3,
                "refit_frequency": 1,
            },
        }
    )
    manifest = build_split_manifest(features, config)
    assert len(manifest.train_target_dates) == 3
    assert len(manifest.validation_target_dates) == 2
    assert len(manifest.test_target_dates) == 2


def test_diebold_mariano_runs() -> None:
    stat, p_value = diebold_mariano(np.array([1.0, 1.1, 0.9]), np.array([1.2, 1.3, 1.1]))
    assert np.isfinite(stat)
    assert 0.0 <= p_value <= 1.0


def test_reconstructor_penalties_zero_for_constant_network() -> None:
    network = ReconstructorNetwork()
    for parameter in network.parameters():
        parameter.data.zero_()
    device = torch.device("cpu")
    surfaces = torch.ones((1, 154), dtype=torch.float32, device=device)
    penalty_grid = build_penalty_grid()
    lc3, lc4, lc5 = _penalty_terms(network, surfaces, penalty_grid, device)
    assert torch.isclose(lc3, torch.tensor(0.0))
    assert torch.isclose(lc4, torch.tensor(0.0))
    assert torch.isclose(lc5, torch.tensor(0.0))
