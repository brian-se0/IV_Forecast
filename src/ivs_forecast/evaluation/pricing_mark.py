from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    forward_price: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    discount_factor: np.ndarray,
    option_type: np.ndarray,
) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    tau = np.maximum(tau, 1e-8)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(forward_price / strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    call_price = discount_factor * (forward_price * norm.cdf(d1) - strike * norm.cdf(d2))
    put_price = discount_factor * (strike * norm.cdf(-d2) - forward_price * norm.cdf(-d1))
    return np.where(option_type == "C", call_price, put_price)


def pricing_utility(
    predicted_price: np.ndarray,
    realized_mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
) -> dict[str, float]:
    inside_spread = (predicted_price >= bid) & (predicted_price <= ask)
    return {
        "price_rmse": float(np.sqrt(np.mean((predicted_price - realized_mid) ** 2))),
        "price_mae": float(np.mean(np.abs(predicted_price - realized_mid))),
        "inside_spread_rate": float(inside_spread.mean()),
    }
