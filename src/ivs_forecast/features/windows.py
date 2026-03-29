from __future__ import annotations

import numpy as np


def trailing_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    result = np.full_like(matrix, np.nan, dtype=np.float64)
    for index in range(window - 1, matrix.shape[0]):
        result[index] = matrix[index - window + 1 : index + 1].mean(axis=0)
    return result
