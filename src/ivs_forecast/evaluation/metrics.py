from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    rmse_iv: float
    mae_iv: float
    mape_iv_clipped: float
    vega_rmse_iv: float


def compute_metrics(
    actual: np.ndarray, predicted: np.ndarray, vegas: np.ndarray
) -> ForecastMetrics:
    errors = predicted - actual
    clipped_denom = np.maximum(actual, 1e-4)
    normalized_vega = vegas / np.maximum(vegas.sum(), 1e-12)
    return ForecastMetrics(
        rmse_iv=float(np.sqrt(np.mean(errors**2))),
        mae_iv=float(np.mean(np.abs(errors))),
        mape_iv_clipped=float(np.mean(np.abs(errors) / clipped_denom)),
        vega_rmse_iv=float(np.sqrt(np.sum(normalized_vega * (errors**2)))),
    )
