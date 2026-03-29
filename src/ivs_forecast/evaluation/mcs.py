from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class McsResult:
    method: str
    included_models: list[str]
    excluded_models: list[str]
    p_values: dict[str, float]


def stationary_bootstrap_indices(
    length: int, average_block_length: int, draws: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / average_block_length
    result = np.empty((draws, length), dtype=np.int64)
    for draw in range(draws):
        index = rng.integers(0, length)
        for t in range(length):
            if t == 0 or rng.random() < p:
                index = rng.integers(0, length)
            else:
                index = (index + 1) % length
            result[draw, t] = index
    return result


def _bootstrap_test_stat(
    loss_matrix: np.ndarray, resampled_indices: np.ndarray, method: str
) -> np.ndarray:
    centered = loss_matrix - loss_matrix.mean(axis=0, keepdims=True)
    boot_means = centered[resampled_indices].mean(axis=1)
    if method == "Tmax":
        pairwise = []
        for i in range(loss_matrix.shape[1]):
            for j in range(i + 1, loss_matrix.shape[1]):
                diff = boot_means[:, i] - boot_means[:, j]
                scale = max(np.std(diff, ddof=1), 1e-12)
                pairwise.append(np.abs(diff / scale))
        return np.max(np.column_stack(pairwise), axis=1)
    if method == "TR":
        relative = boot_means - boot_means.min(axis=1, keepdims=True)
        scale = np.maximum(relative.std(axis=0, ddof=1), 1e-12)
        return np.max(relative / scale, axis=1)
    raise ValueError(f"Unknown MCS method: {method}")


def _observed_stat(loss_matrix: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    mean_losses = loss_matrix.mean(axis=0)
    if method == "Tmax":
        stats = []
        for i in range(loss_matrix.shape[1]):
            for j in range(i + 1, loss_matrix.shape[1]):
                diff = loss_matrix[:, i] - loss_matrix[:, j]
                scale = max(diff.std(ddof=1), 1e-12)
                stats.append(abs(diff.mean()) / scale)
        array = np.array(stats, dtype=np.float64)
        return array, float(array.max())
    if method == "TR":
        centered = mean_losses - mean_losses.min()
        scale = np.maximum(loss_matrix.std(axis=0, ddof=1), 1e-12)
        array = centered / scale
        return array, float(array.max())
    raise ValueError(f"Unknown MCS method: {method}")


def run_mcs(
    loss_by_model: dict[str, np.ndarray],
    alpha: float = 0.10,
    bootstrap_draws: int = 5000,
    block_length: int = 10,
    seed: int = 20260329,
) -> dict[str, Any]:
    model_names = list(sorted(loss_by_model))
    loss_matrix = np.column_stack([loss_by_model[name] for name in model_names])
    output: dict[str, Any] = {}
    bootstrap_indices = stationary_bootstrap_indices(
        loss_matrix.shape[0], block_length, bootstrap_draws, seed
    )
    for method in ("Tmax", "TR"):
        active_models = model_names[:]
        p_values: dict[str, float] = {}
        while len(active_models) > 1:
            active_loss = np.column_stack([loss_by_model[name] for name in active_models])
            observed_per_model, observed = _observed_stat(active_loss, method)
            boot = _bootstrap_test_stat(active_loss, bootstrap_indices, method)
            p_value = float(np.mean(boot >= observed))
            worst_model_index = int(np.argmax(active_loss.mean(axis=0)))
            p_values[active_models[worst_model_index]] = p_value
            if p_value > alpha:
                break
            del active_models[worst_model_index]
        included = active_models
        excluded = [name for name in model_names if name not in included]
        output[method] = McsResult(
            method=method,
            included_models=included,
            excluded_models=excluded,
            p_values=p_values,
        ).__dict__
    return output
