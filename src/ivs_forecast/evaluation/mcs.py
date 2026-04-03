from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class McsBootstrapSettings:
    method: str
    alpha: float
    bootstrap: str
    draws: int
    block_length: int
    seed: int


def stationary_bootstrap_indices(
    length: int, average_block_length: int, draws: int, seed: int
) -> np.ndarray:
    if length < 2:
        raise ValueError("MCS requires at least two observations per loss series.")
    if average_block_length <= 0:
        raise ValueError("MCS stationary bootstrap block length must be positive.")
    if draws <= 0:
        raise ValueError("MCS stationary bootstrap requires at least one draw.")
    rng = np.random.default_rng(seed)
    continuation_probability = 1.0 - (1.0 / average_block_length)
    result = np.empty((draws, length), dtype=np.int64)
    for draw in range(draws):
        index = rng.integers(0, length)
        for offset in range(length):
            if offset == 0 or rng.random() > continuation_probability:
                index = rng.integers(0, length)
            else:
                index = (index + 1) % length
            result[draw, offset] = index
    return result


def _validated_loss_matrix(loss_by_model: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    if len(loss_by_model) < 2:
        raise ValueError("MCS requires at least two model loss series.")
    model_names = list(sorted(loss_by_model))
    arrays: list[np.ndarray] = []
    expected_length: int | None = None
    for model_name in model_names:
        array = np.asarray(loss_by_model[model_name], dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"MCS loss series for {model_name} must be one-dimensional.")
        if not np.isfinite(array).all():
            raise ValueError(f"MCS loss series for {model_name} contains non-finite values.")
        if expected_length is None:
            expected_length = array.shape[0]
        elif array.shape[0] != expected_length:
            raise ValueError("All MCS loss series must have equal length.")
        if array.shape[0] < 2:
            raise ValueError("All MCS loss series must contain at least two observations.")
        arrays.append(array)
    return model_names, np.column_stack(arrays)


def _pairwise_mean_differentials(mean_losses: np.ndarray) -> np.ndarray:
    return mean_losses[:, np.newaxis] - mean_losses[np.newaxis, :]


def _bootstrap_statistics(
    loss_matrix: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed_mean = loss_matrix.mean(axis=0)
    observed_dbar_ij = _pairwise_mean_differentials(observed_mean)
    bootstrap_means = loss_matrix[bootstrap_indices].mean(axis=1)
    bootstrap_dbar_ij = bootstrap_means[:, :, np.newaxis] - bootstrap_means[:, np.newaxis, :]
    centered_bootstrap_dbar_ij = bootstrap_dbar_ij - observed_dbar_ij[np.newaxis, :, :]
    variance_dbar_ij = np.maximum(centered_bootstrap_dbar_ij.var(axis=0, ddof=1), 1e-12)
    model_count = loss_matrix.shape[1]
    observed_dbar_i = observed_dbar_ij.sum(axis=1) / (model_count - 1)
    centered_bootstrap_dbar_i = centered_bootstrap_dbar_ij.sum(axis=2) / (model_count - 1)
    variance_dbar_i = np.maximum(centered_bootstrap_dbar_i.var(axis=0, ddof=1), 1e-12)
    return observed_dbar_ij, variance_dbar_ij, observed_dbar_i, variance_dbar_i


def _step_result(
    active_models: list[str],
    loss_matrix: np.ndarray,
    method: str,
    bootstrap_indices: np.ndarray,
) -> tuple[float, float, str]:
    observed_dbar_ij, variance_dbar_ij, observed_dbar_i, variance_dbar_i = _bootstrap_statistics(
        loss_matrix,
        bootstrap_indices,
    )
    observed_t_ij = observed_dbar_ij / np.sqrt(variance_dbar_ij)
    observed_t_i = observed_dbar_i / np.sqrt(variance_dbar_i)
    bootstrap_means = loss_matrix[bootstrap_indices].mean(axis=1)
    bootstrap_dbar_ij = bootstrap_means[:, :, np.newaxis] - bootstrap_means[:, np.newaxis, :]
    centered_bootstrap_dbar_ij = bootstrap_dbar_ij - observed_dbar_ij[np.newaxis, :, :]
    centered_bootstrap_t_ij = centered_bootstrap_dbar_ij / np.sqrt(variance_dbar_ij)
    centered_bootstrap_dbar_i = centered_bootstrap_dbar_ij.sum(axis=2) / (loss_matrix.shape[1] - 1)
    centered_bootstrap_t_i = centered_bootstrap_dbar_i / np.sqrt(variance_dbar_i)

    if method == "Tmax":
        observed_statistic = float(observed_t_i.max())
        bootstrap_statistic = centered_bootstrap_t_i.max(axis=1)
        eliminated_model = active_models[int(observed_t_i.argmax())]
    elif method == "TR":
        upper_triangle = np.triu_indices(loss_matrix.shape[1], k=1)
        observed_statistic = float(np.abs(observed_t_ij[upper_triangle]).max())
        bootstrap_statistic = np.abs(centered_bootstrap_t_ij[:, upper_triangle[0], upper_triangle[1]]).max(
            axis=1
        )
        eliminated_model = active_models[int(observed_t_ij.max(axis=1).argmax())]
    else:
        raise ValueError(f"Unknown MCS method: {method}")
    p_value = float(np.mean(bootstrap_statistic >= observed_statistic))
    return observed_statistic, p_value, eliminated_model


def _run_single_method(
    model_names: list[str],
    full_loss_matrix: np.ndarray,
    method: str,
    alpha: float,
    bootstrap_indices: np.ndarray,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    active_models = model_names[:]
    active_indices = list(range(len(model_names)))
    elimination_order: list[str] = []
    p_values_by_step: list[dict[str, Any]] = []
    step = 1
    while len(active_models) > 1:
        loss_matrix = full_loss_matrix[:, active_indices]
        observed_statistic, p_value, eliminated_model = _step_result(
            active_models,
            loss_matrix,
            method,
            bootstrap_indices,
        )
        p_values_by_step.append(
            {
                "step": step,
                "models": active_models[:],
                "observed_statistic": observed_statistic,
                "p_value": p_value,
                "eliminated_model": None if p_value > alpha else eliminated_model,
            }
        )
        if p_value > alpha:
            break
        eliminated_index = active_models.index(eliminated_model)
        elimination_order.append(eliminated_model)
        del active_models[eliminated_index]
        del active_indices[eliminated_index]
        step += 1
    excluded_models = [name for name in model_names if name not in active_models]
    return {
        "method": method,
        "alpha": alpha,
        "bootstrap_settings": McsBootstrapSettings(
            method=method,
            alpha=alpha,
            bootstrap="stationary",
            draws=int(bootstrap_indices.shape[0]),
            block_length=block_length,
            seed=seed,
        ).__dict__,
        "elimination_order": elimination_order,
        "included_models": active_models,
        "excluded_models": excluded_models,
        "p_values_by_step": p_values_by_step,
    }


def run_mcs(
    loss_by_model: dict[str, np.ndarray],
    alpha: float = 0.10,
    bootstrap_draws: int = 5000,
    block_length: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    model_names, loss_matrix = _validated_loss_matrix(loss_by_model)
    bootstrap_indices = stationary_bootstrap_indices(
        loss_matrix.shape[0], block_length, bootstrap_draws, seed
    )
    return {
        method: _run_single_method(
            model_names=model_names,
            full_loss_matrix=loss_matrix,
            method=method,
            alpha=alpha,
            bootstrap_indices=bootstrap_indices,
            block_length=block_length,
            seed=seed,
        )
        for method in ("Tmax", "TR")
    }
