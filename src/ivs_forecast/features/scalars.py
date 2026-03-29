from __future__ import annotations

import numpy as np


def log_return(series: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full(series.shape, np.nan, dtype=np.float64)
    for index in range(horizon, series.shape[0]):
        result[index] = np.log(series[index] / series[index - horizon])
    return result
