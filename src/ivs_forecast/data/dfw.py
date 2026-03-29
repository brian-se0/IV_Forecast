from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


def default_grid_definition() -> pl.DataFrame:
    moneyness_ratios = [
        0.60,
        0.80,
        0.90,
        0.95,
        0.975,
        1.00,
        1.025,
        1.05,
        1.10,
        1.20,
        1.30,
        1.50,
        1.75,
        2.00,
    ]
    maturities_days = [10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730]
    records: list[dict[str, float | int | str]] = []
    index = 0
    for ratio in moneyness_ratios:
        for maturity_days in maturities_days:
            records.append(
                {
                    "grid_id": f"g{index:03d}",
                    "grid_index": index,
                    "m": float(np.log(ratio)),
                    "tau": maturity_days / 365.0,
                    "moneyness_ratio": ratio,
                    "maturity_days": maturity_days,
                }
            )
            index += 1
    return pl.DataFrame(records)


@dataclass(frozen=True)
class DfwFitResult:
    quote_date: object
    coefficients: np.ndarray
    fitted_values: np.ndarray


def fit_dfw_surface(nodes: pl.DataFrame) -> DfwFitResult:
    m = nodes["m"].to_numpy()
    tau = nodes["tau"].to_numpy()
    y = nodes["node_iv"].to_numpy()
    weights = nodes["node_vega"].to_numpy()
    design = np.column_stack([np.ones_like(m), m, tau, m**2, tau**2, m * tau])
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = y * sqrt_weights
    coeffs, residuals, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(f"Singular or rank-deficient DFW design for {nodes['quote_date'][0]}")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError(f"Non-finite DFW coefficients for {nodes['quote_date'][0]}")
    fitted = np.maximum(0.01, design @ coeffs)
    if residuals.size and not np.all(np.isfinite(residuals)):
        raise ValueError(f"Invalid DFW residuals for {nodes['quote_date'][0]}")
    return DfwFitResult(
        quote_date=nodes["quote_date"][0], coefficients=coeffs, fitted_values=fitted
    )


def sample_dfw(coefficients: np.ndarray, grid_definition: pl.DataFrame) -> np.ndarray:
    m = grid_definition["m"].to_numpy()
    tau = grid_definition["tau"].to_numpy()
    design = np.column_stack([np.ones_like(m), m, tau, m**2, tau**2, m * tau])
    return np.maximum(0.01, design @ coefficients)
