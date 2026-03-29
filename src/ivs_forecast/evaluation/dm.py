from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class DieboldMarianoResult:
    model_a: str
    model_b: str
    statistic: float
    p_value: float
    adjusted_p_value: float | None = None


def _newey_west_variance(differential: np.ndarray, bandwidth: int) -> float:
    centered = differential - differential.mean()
    gamma0 = np.dot(centered, centered) / centered.shape[0]
    variance = gamma0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = np.dot(centered[lag:], centered[:-lag]) / centered.shape[0]
        variance += 2.0 * weight * gamma
    return float(max(variance, 1e-12))


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    bandwidth: int = 5,
    horizon: int = 1,
) -> tuple[float, float]:
    differential = np.asarray(loss_a - loss_b, dtype=np.float64)
    t = differential.shape[0]
    variance = _newey_west_variance(differential, bandwidth) / t
    statistic = differential.mean() / np.sqrt(variance)
    hln = np.sqrt((t + 1.0 - 2.0 * horizon + (horizon * (horizon - 1.0)) / t) / t)
    adjusted = statistic * hln
    p_value = 2.0 * (1.0 - norm.cdf(abs(adjusted)))
    return float(adjusted), float(p_value)


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(indexed, start=1):
        candidate = min(1.0, (total - rank + 1) * p_value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def pairwise_dm(loss_by_model: dict[str, np.ndarray]) -> list[DieboldMarianoResult]:
    raw_results: list[DieboldMarianoResult] = []
    p_values: list[float] = []
    pairs = list(combinations(sorted(loss_by_model), 2))
    for model_a, model_b in pairs:
        statistic, p_value = diebold_mariano(loss_by_model[model_a], loss_by_model[model_b])
        raw_results.append(
            DieboldMarianoResult(
                model_a=model_a,
                model_b=model_b,
                statistic=statistic,
                p_value=p_value,
            )
        )
        p_values.append(p_value)
    adjusted = holm_adjust(p_values)
    return [
        DieboldMarianoResult(
            model_a=result.model_a,
            model_b=result.model_b,
            statistic=result.statistic,
            p_value=result.p_value,
            adjusted_p_value=adjusted[index],
        )
        for index, result in enumerate(raw_results)
    ]
