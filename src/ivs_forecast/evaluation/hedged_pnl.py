from __future__ import annotations

import numpy as np


def hedged_pnl_utility(
    mid_t: np.ndarray,
    mid_t1: np.ndarray,
    predicted_price_t1: np.ndarray,
    delta_t: np.ndarray,
    underlying_t: np.ndarray,
    underlying_t1: np.ndarray,
) -> dict[str, float]:
    realized = (mid_t1 - mid_t) - delta_t * (underlying_t1 - underlying_t)
    predicted = (predicted_price_t1 - mid_t) - delta_t * (underlying_t1 - underlying_t)
    return {
        "hedged_pnl_rmse": float(np.sqrt(np.mean((predicted - realized) ** 2))),
        "hedged_pnl_mae": float(np.mean(np.abs(predicted - realized))),
    }
