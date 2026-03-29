from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class DieboldMarianoResult:
    model_a: str
    model_b: str
    statistic: float
    p_value: float
    adjusted_p_value: float | None = None
    n_obs: int = 0
    bandwidth: int = 5
    horizon: int = 1


def _validated_loss_pair(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    bandwidth: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    series_a = np.asarray(loss_a, dtype=np.float64)
    series_b = np.asarray(loss_b, dtype=np.float64)
    if series_a.ndim != 1 or series_b.ndim != 1:
        raise ValueError("Diebold-Mariano loss inputs must be one-dimensional.")
    if series_a.shape[0] != series_b.shape[0]:
        raise ValueError("Diebold-Mariano loss series must have equal length.")
    if series_a.shape[0] < max(3, horizon + 1):
        raise ValueError("Diebold-Mariano requires at least three aligned observations.")
    if bandwidth < 0:
        raise ValueError("Diebold-Mariano bandwidth must be non-negative.")
    if horizon < 1:
        raise ValueError("Diebold-Mariano horizon must be positive.")
    if not np.isfinite(series_a).all() or not np.isfinite(series_b).all():
        raise ValueError("Diebold-Mariano loss series must contain only finite numeric values.")
    return series_a, series_b


def _newey_west_long_run_variance(differential: np.ndarray, bandwidth: int) -> float:
    centered = differential - differential.mean()
    sample_size = centered.shape[0]
    gamma0 = np.dot(centered, centered) / sample_size
    variance = gamma0
    max_lag = min(bandwidth, sample_size - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.dot(centered[lag:], centered[:-lag]) / sample_size
        variance += 2.0 * weight * gamma
    return float(max(variance, 1e-12))


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    bandwidth: int = 5,
    horizon: int = 1,
) -> tuple[float, float]:
    series_a, series_b = _validated_loss_pair(loss_a, loss_b, bandwidth, horizon)
    differential = series_a - series_b
    sample_size = differential.shape[0]
    long_run_variance = _newey_west_long_run_variance(differential, bandwidth) / sample_size
    raw_statistic = differential.mean() / np.sqrt(long_run_variance)
    hln_scale = np.sqrt(
        (sample_size + 1.0 - 2.0 * horizon + (horizon * (horizon - 1.0)) / sample_size)
        / sample_size
    )
    adjusted_statistic = raw_statistic * hln_scale
    p_value = 2.0 * (1.0 - student_t.cdf(abs(adjusted_statistic), df=sample_size - 1))
    return float(adjusted_statistic), float(p_value)


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


def _validated_loss_by_model(loss_by_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(loss_by_model) < 2:
        raise ValueError("Diebold-Mariano requires at least two model loss series.")
    validated: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for model_name, values in sorted(loss_by_model.items()):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"Loss series for {model_name} must be one-dimensional.")
        if not np.isfinite(array).all():
            raise ValueError(f"Loss series for {model_name} contains non-finite values.")
        if expected_length is None:
            expected_length = array.shape[0]
        elif array.shape[0] != expected_length:
            raise ValueError("All model loss series must share the same number of observations.")
        if array.shape[0] < 3:
            raise ValueError("All model loss series must contain at least three observations.")
        validated[model_name] = array
    return validated


def pairwise_dm(
    loss_by_model: dict[str, np.ndarray],
    bandwidth: int = 5,
    horizon: int = 1,
) -> list[DieboldMarianoResult]:
    validated = _validated_loss_by_model(loss_by_model)
    raw_results: list[DieboldMarianoResult] = []
    p_values: list[float] = []
    pairs = list(combinations(sorted(validated), 2))
    for model_a, model_b in pairs:
        statistic, p_value = diebold_mariano(
            validated[model_a],
            validated[model_b],
            bandwidth=bandwidth,
            horizon=horizon,
        )
        raw_results.append(
            DieboldMarianoResult(
                model_a=model_a,
                model_b=model_b,
                statistic=statistic,
                p_value=p_value,
                n_obs=int(validated[model_a].shape[0]),
                bandwidth=bandwidth,
                horizon=horizon,
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
            n_obs=result.n_obs,
            bandwidth=result.bandwidth,
            horizon=result.horizon,
        )
        for index, result in enumerate(raw_results)
    ]
