from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ForwardEstimationDiagnostics:
    quote_date: str
    expiration: str
    matched_pairs_before_prune: int
    matched_pairs_after_prune: int
    invalid_reason: str | None


def _weighted_linear_fit(
    strikes: np.ndarray, y_values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(strikes), strikes])
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = y_values * sqrt_weights
    coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def estimate_forward_terms(
    clean_contracts: pl.DataFrame,
) -> tuple[pl.DataFrame, list[ForwardEstimationDiagnostics]]:
    records: list[dict[str, object]] = []
    diagnostics: list[ForwardEstimationDiagnostics] = []
    for (quote_date, expiration), group in clean_contracts.partition_by(
        ["quote_date", "expiration"], as_dict=True
    ).items():
        calls = group.filter(pl.col("option_type") == "C").select(
            "strike", pl.col("mid_1545").alias("call_mid"), pl.col("vega_1545").alias("call_vega")
        )
        puts = group.filter(pl.col("option_type") == "P").select(
            "strike", pl.col("mid_1545").alias("put_mid"), pl.col("vega_1545").alias("put_vega")
        )
        matched = calls.join(puts, on="strike", how="inner").sort("strike")
        before = matched.height
        invalid_reason: str | None = None
        if before < 3:
            diagnostics.append(
                ForwardEstimationDiagnostics(
                    quote_date=str(quote_date),
                    expiration=str(expiration),
                    matched_pairs_before_prune=before,
                    matched_pairs_after_prune=0,
                    invalid_reason="fewer_than_3_matched_strikes",
                )
            )
            continue
        strikes = matched["strike"].to_numpy()
        y_values = (matched["call_mid"] - matched["put_mid"]).to_numpy()
        weights = np.minimum(matched["call_vega"].to_numpy(), matched["put_vega"].to_numpy())
        alpha, beta = _weighted_linear_fit(strikes, y_values, weights)
        residuals = y_values - (alpha + beta * strikes)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad > 0:
            keep = np.abs(residuals) <= 5.0 * mad
        else:
            keep = np.ones_like(residuals, dtype=bool)
        strikes_refit = strikes[keep]
        y_refit = y_values[keep]
        weights_refit = weights[keep]
        after = int(keep.sum())
        if after < 3:
            invalid_reason = "fewer_than_3_after_mad_prune"
        else:
            alpha, beta = _weighted_linear_fit(strikes_refit, y_refit, weights_refit)
            discount_factor = -beta
            forward_price = alpha / discount_factor if discount_factor != 0 else np.nan
            if not np.isfinite(discount_factor) or discount_factor <= 0:
                invalid_reason = "non_positive_discount_factor"
            elif not np.isfinite(forward_price) or forward_price <= 0:
                invalid_reason = "non_positive_forward_price"
            else:
                records.append(
                    {
                        "quote_date": quote_date,
                        "expiration": expiration,
                        "discount_factor": float(discount_factor),
                        "forward_price": float(forward_price),
                        "matched_pairs_before_prune": before,
                        "matched_pairs_after_prune": after,
                    }
                )
        diagnostics.append(
            ForwardEstimationDiagnostics(
                quote_date=str(quote_date),
                expiration=str(expiration),
                matched_pairs_before_prune=before,
                matched_pairs_after_prune=after,
                invalid_reason=invalid_reason,
            )
        )
    return pl.DataFrame(records).sort(["quote_date", "expiration"]), diagnostics
