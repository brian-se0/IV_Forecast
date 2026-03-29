from __future__ import annotations

import numpy as np


def straddle_utility(net_returns: np.ndarray, gross_returns: np.ndarray) -> dict[str, float]:
    sharpe = 0.0
    if net_returns.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252.0) * net_returns.mean() / net_returns.std(ddof=1))
    return {
        "mean_gross_return": float(gross_returns.mean()),
        "mean_net_return": float(net_returns.mean()),
        "hit_rate": float((net_returns > 0).mean()),
        "sharpe_ratio": sharpe,
    }
