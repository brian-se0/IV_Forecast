from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from ivs_forecast.config import AppConfig
from ivs_forecast.data.ssvi import raw_to_constrained_params, ssvi_implied_vol
from ivs_forecast.evaluation.dm import diebold_mariano
from ivs_forecast.features.dataset import build_features_targets
from ivs_forecast.features.scalars import add_state_scalar_features
from ivs_forecast.models.base import (
    DailyStateStore,
    fit_normalization,
    history_feature_columns,
    unique_history_indices,
)
from ivs_forecast.models.ssvi_tcn_direct import (
    SsviTcnNetwork,
    masked_vega_huber_loss,
)
from ivs_forecast.models.state_var1 import StateVar1Model
from ivs_forecast.pipeline.splits import assert_refit_window_precedes_chunk, build_split_manifest


def _ssvi_state_panel(n_dates: int = 32) -> pl.DataFrame:
    start_date = date(2020, 1, 2)
    rows = []
    for index in range(n_dates):
        quote_date = start_date + timedelta(days=index)
        theta = np.linspace(0.02, 0.06, 11) + 0.0003 * index
        raw_state = np.concatenate(
            [
                np.log(theta),
                np.asarray([-0.35 + 0.001 * index, -0.25, 0.0], dtype=np.float64),
            ]
        )
        constrained = np.asarray(raw_to_constrained_params(raw_state), dtype=np.float64)
        row = {
            "state_row_index": index,
            "quote_date": quote_date,
            "option_root": "SPX",
            "active_underlying_price_1545": 3000.0 + index,
            "total_trade_volume": 10_000 + 10 * index,
            "total_open_interest": 20_000 + 20 * index,
            "median_rel_spread_1545": 0.01 + 1e-4 * index,
            "surface_fit_rmse_iv": 0.005 + 1e-4 * index,
            "surface_fit_vega_rmse_iv": 0.004 + 1e-4 * index,
            "surface_node_count": 60,
            "valid_expiry_count": 5,
        }
        for raw_index, value in enumerate(raw_state):
            row[f"state_z_{raw_index:03d}"] = float(value)
        columns = [f"theta_d{day:03d}" for day in (10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730)]
        for column, value in zip(columns, constrained[:11], strict=True):
            row[column] = float(value)
        row["rho"] = float(constrained[11])
        row["eta"] = float(constrained[12])
        row["lambda"] = float(constrained[13])
        rows.append(row)
    return add_state_scalar_features(pl.DataFrame(rows))


def test_build_features_targets_and_split_manifest() -> None:
    state_panel = _ssvi_state_panel(32)
    features = build_features_targets(state_panel)
    assert features.height == 10
    assert features["quote_date"][0] == state_panel["quote_date"][21]
    assert features["target_date"][0] == state_panel["quote_date"][22]
    config = AppConfig.model_validate(
        {
            "paths": {"raw_data_root": "D:/Options Data", "artifact_root": "./artifacts"},
            "study": {
                "underlying_symbol": "^SPX",
                "option_root": "SPX",
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "forecast_horizon_days": 1,
            },
            "split": {
                "validation_size": 2,
                "test_size": 2,
                "min_train_size": 6,
                "refit_frequency": 1,
            },
        }
    )
    manifest = build_split_manifest(features, config)
    assert len(manifest.train_target_dates) == 6
    assert len(manifest.validation_target_dates) == 2
    assert len(manifest.test_target_dates) == 2


def test_future_shock_does_not_enter_origin_feature_rows() -> None:
    baseline_panel = _ssvi_state_panel(28)
    shocked_panel = baseline_panel.with_columns(
        pl.when(pl.col("state_row_index") >= 25)
        .then(pl.lit(1_000_000.0))
        .otherwise(pl.col("total_trade_volume"))
        .alias("total_trade_volume")
    )
    shocked_panel = add_state_scalar_features(shocked_panel)
    baseline_features = build_features_targets(baseline_panel)
    shocked_features = build_features_targets(shocked_panel)
    assert shocked_features["log_total_trade_volume"][0] == pytest.approx(
        baseline_features["log_total_trade_volume"][0]
    )


def test_normalization_fit_only_on_training_rows() -> None:
    state_panel = _ssvi_state_panel(35).with_columns(
        pl.when(pl.col("state_row_index") >= 30)
        .then(pl.lit(25.0))
        .otherwise(pl.col("surface_fit_rmse_iv"))
        .alias("surface_fit_rmse_iv")
    )
    state_store = DailyStateStore.from_frame(state_panel)
    features = build_features_targets(state_panel)
    train_rows = features.head(6)
    stats = fit_normalization(train_rows, state_store, history_days=10)
    expected_indices = unique_history_indices(train_rows, history_days=10)
    expected_matrix = state_store.daily_feature_matrix[expected_indices].astype(np.float64)
    assert np.allclose(stats.mean, expected_matrix.mean(axis=0))
    assert stats.mean[-3] < 1.0


def test_state_var1_artifact_shape() -> None:
    state_panel = _ssvi_state_panel(30)
    state_store = DailyStateStore.from_frame(state_panel)
    features = build_features_targets(state_panel)
    model = StateVar1Model()
    model.fit(features.head(6), state_store)
    prediction = model.predict(features.slice(6, 1), state_store)
    assert prediction.shape == (1, 14)
    artifact = model.artifact()
    assert artifact.params["intercept_shape"] == [14]
    assert artifact.params["transition_shape"] == [14, 14]


def test_diebold_mariano_runs() -> None:
    stat, p_value = diebold_mariano(np.array([1.0, 1.1, 0.9]), np.array([1.2, 1.3, 1.1]))
    assert np.isfinite(stat)
    assert 0.0 <= p_value <= 1.0


def test_diebold_mariano_matches_regression_fixture() -> None:
    loss_a = np.array([0.80, 1.00, 1.10, 0.95, 1.05, 0.90, 1.15, 0.98], dtype=np.float64)
    loss_b = np.array([1.05, 1.10, 1.20, 1.08, 1.16, 1.02, 1.22, 1.09], dtype=np.float64)
    stat, p_value = diebold_mariano(loss_a, loss_b, bandwidth=2, horizon=1)
    assert stat == pytest.approx(-7.148256223078783)
    assert p_value == pytest.approx(0.00018564281821276118)


def test_diebold_mariano_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        diebold_mariano(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_refit_boundary_guard() -> None:
    available_rows = pl.DataFrame(
        {
            "quote_date": [date(2020, 1, 2), date(2020, 1, 3)],
            "target_date": [date(2020, 1, 3), date(2020, 1, 6)],
        }
    )
    chunk_rows = pl.DataFrame(
        {
            "quote_date": [date(2020, 1, 6)],
            "target_date": [date(2020, 1, 7)],
        }
    )
    assert_refit_window_precedes_chunk(available_rows, chunk_rows, "unit_test_refit")
    with pytest.raises(ValueError, match="chronology violation"):
        assert_refit_window_precedes_chunk(
            available_rows.vstack(chunk_rows), chunk_rows, "unit_test_refit"
        )


def test_ssvi_tcn_forward_pass_and_masked_loss() -> None:
    network = SsviTcnNetwork(input_dim=len(history_feature_columns()), width=32, dropout=0.1)
    history = torch.randn(4, 10, len(history_feature_columns()))
    predicted_state = network(history)
    constrained = raw_to_constrained_params(predicted_state)
    m = torch.zeros(4, 3)
    tau = torch.full((4, 3), 30.0 / 365.0)
    target_iv = torch.full((4, 3), 0.20)
    target_vega = torch.ones(4, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]] * 4)
    predicted_iv = ssvi_implied_vol(m, tau, constrained)
    loss = masked_vega_huber_loss(predicted_iv, target_iv, target_vega, mask)
    assert predicted_state.shape == (4, 14)
    assert torch.isfinite(loss)


def test_synthetic_easy_panel_loss_decreases() -> None:
    torch.manual_seed(7)
    network = SsviTcnNetwork(input_dim=len(history_feature_columns()), width=32, dropout=0.0)
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-3)
    history = torch.randn(16, 10, len(history_feature_columns()))
    target_state = history[:, -1, :14]
    target_params = raw_to_constrained_params(target_state)
    m = torch.tensor([[0.0, 0.05, -0.05]] * 16)
    tau = torch.tensor([[30.0 / 365.0, 60.0 / 365.0, 91.0 / 365.0]] * 16)
    target_iv = ssvi_implied_vol(m, tau, target_params).detach()
    target_vega = torch.ones_like(target_iv)
    mask = torch.ones_like(target_iv)
    first_loss = None
    last_loss = None
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        predicted_state = network(history)
        predicted_iv = ssvi_implied_vol(m, tau, raw_to_constrained_params(predicted_state))
        loss = masked_vega_huber_loss(predicted_iv, target_iv, target_vega, mask)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    assert last_loss is not None and first_loss is not None
    assert last_loss < first_loss
