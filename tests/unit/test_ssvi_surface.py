from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivs_forecast.data.ssvi import (
    SsviCalibrationConfig,
    calibrate_daily_ssvi,
    constrained_to_raw_params,
    raw_to_constrained_params,
    ssvi_implied_vol,
    static_arb_certification,
)


def _synthetic_nodes(params: np.ndarray) -> pl.DataFrame:
    m_grid = np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20], dtype=np.float64)
    tau_grid = np.array([10, 30, 60, 91, 182, 365, 730], dtype=np.float64) / 365.0
    rows = []
    for tau in tau_grid:
        for strike_index, moneyness in enumerate(m_grid):
            rows.append(
                {
                    "quote_date": date(2020, 1, 2),
                    "option_root": "SPX",
                    "root": "SPX",
                    "expiration": date(2020, 2, 21),
                    "strike": 3000.0 + 5.0 * strike_index,
                    "m": moneyness,
                    "tau": tau,
                    "node_iv": float(
                        np.asarray(
                            ssvi_implied_vol(
                                np.asarray([moneyness], dtype=np.float64),
                                np.asarray([tau], dtype=np.float64),
                                params,
                            ),
                            dtype=np.float64,
                        )[0]
                    ),
                    "node_vega": 1.0 + 0.1 * strike_index,
                    "active_underlying_price_1545": 3000.0,
                    "trade_volume": 100,
                    "open_interest": 1000,
                    "median_rel_spread_1545": 0.01,
                }
            )
    return pl.DataFrame(rows)


def test_synthetic_parameter_recovery() -> None:
    constrained = np.asarray(
        raw_to_constrained_params(
            np.asarray(
                [
                    *np.log(np.linspace(0.02, 0.05, 11)),
                    -0.4,
                    -0.8,
                    0.2,
                ],
                dtype=np.float64,
            )
        ),
        dtype=np.float64,
    )
    nodes = _synthetic_nodes(constrained)
    result = calibrate_daily_ssvi(nodes, init_raw=None, config=SsviCalibrationConfig())
    fitted_iv = np.asarray(
        ssvi_implied_vol(
            nodes["m"].to_numpy().astype(np.float64),
            nodes["tau"].to_numpy().astype(np.float64),
            result.constrained_params,
        ),
        dtype=np.float64,
    )
    actual_iv = nodes["node_iv"].to_numpy().astype(np.float64)
    assert np.sqrt(np.mean((fitted_iv - actual_iv) ** 2)) < 2e-3


def test_certification_passes_generated_surface() -> None:
    constrained = np.asarray(
        raw_to_constrained_params(
            np.asarray(
                [
                    *np.log(np.linspace(0.02, 0.05, 11)),
                    -0.35,
                    -1.0,
                    0.1,
                ],
                dtype=np.float64,
            )
        ),
        dtype=np.float64,
    )
    certification = static_arb_certification(constrained, tol=1e-8)
    assert certification["passes_static_arb"] is True
    assert certification["calendar_violation_count"] == 0
    assert certification["butterfly_violation_count"] == 0


def test_decode_to_iv_is_positive_and_constrained() -> None:
    raw = np.asarray([5.0, -3.0, 2.0, -4.0, 1.0, 0.5, -1.5, -2.0, 0.1, 0.2, -0.7, 4.0, 5.0, -5.0])
    constrained = np.asarray(raw_to_constrained_params(raw), dtype=np.float64)
    iv = np.asarray(
        ssvi_implied_vol(
            np.asarray([0.0, 0.10, -0.10], dtype=np.float64),
            np.asarray([30.0 / 365.0, 91.0 / 365.0, 365.0 / 365.0], dtype=np.float64),
            constrained,
        ),
        dtype=np.float64,
    )
    assert np.all(iv > 0.0)
    assert np.all(np.diff(constrained[:11]) >= 0.0)
    assert abs(constrained[11]) < 1.0
    assert constrained[12] > 0.0
    assert 0.0 <= constrained[13] <= 0.5


def test_constrained_to_raw_round_trip_preserves_valid_surface() -> None:
    constrained = np.asarray(
        raw_to_constrained_params(
            np.asarray(
                [
                    *np.log(np.linspace(0.02, 0.04, 11)),
                    -0.25,
                    -0.9,
                    0.3,
                ],
                dtype=np.float64,
            )
        ),
        dtype=np.float64,
    )
    round_trip = np.asarray(raw_to_constrained_params(constrained_to_raw_params(constrained)), dtype=np.float64)
    assert np.allclose(round_trip, constrained)
