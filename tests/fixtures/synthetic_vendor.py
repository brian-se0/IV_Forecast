from __future__ import annotations

import csv
import io
import zipfile
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm

HEADER = [
    "underlying_symbol",
    "quote_date",
    "root",
    "expiration",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "trade_volume",
    "bid_size_1545",
    "bid_1545",
    "ask_size_1545",
    "ask_1545",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "implied_underlying_price_1545",
    "active_underlying_price_1545",
    "implied_volatility_1545",
    "delta_1545",
    "gamma_1545",
    "theta_1545",
    "vega_1545",
    "rho_1545",
    "bid_size_eod",
    "bid_eod",
    "ask_size_eod",
    "ask_eod",
    "underlying_bid_eod",
    "underlying_ask_eod",
    "vwap",
    "open_interest",
    "delivery_code",
]


def _business_dates(start: np.datetime64, count: int) -> list[np.datetime64]:
    dates: list[np.datetime64] = []
    current = start
    while len(dates) < count:
        if np.is_busday(current):
            dates.append(current)
        current = current + np.timedelta64(1, "D")
    return dates


def _bs_values(
    spot: float, strike: float, tau: float, sigma: float, rate: float, option_type: str
) -> tuple[float, float, float, float]:
    discount_factor = exp(-rate * tau)
    forward = spot * exp(rate * tau)
    d1 = (log(forward / strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    call_price = discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    put_price = discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    price = call_price if option_type == "C" else put_price
    delta = norm.cdf(d1) if option_type == "C" else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (spot * sigma * sqrt(tau))
    vega = discount_factor * forward * norm.pdf(d1) * sqrt(tau)
    return price, delta, gamma, vega


def write_synthetic_vendor_dataset(root: Path, n_dates: int = 70) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    dates = _business_dates(np.datetime64("2020-01-02"), n_dates)
    rate = 0.01
    ratios = [0.80, 0.85, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15, 1.20]
    expiry_days = [30, 60, 91, 182, 365]
    for index, np_date in enumerate(dates):
        quote_date = str(np_date.astype("datetime64[D]"))
        spot = 3000.0 + 2.5 * index + 40.0 * np.sin(index / 7.0)
        rows: list[list[str]] = []
        for expiry_offset in expiry_days:
            expiration = str((np_date + np.timedelta64(expiry_offset, "D")).astype("datetime64[D]"))
            tau = expiry_offset / 365.0
            for ratio in ratios:
                strike = round(spot * ratio / 5.0) * 5.0
                m = log(strike / (spot * exp(rate * tau)))
                sigma = 0.14 + 0.03 * abs(m) + 0.015 * tau + 0.01 * np.sin(index / 5.0)
                for option_type in ("C", "P"):
                    price, delta, gamma, vega = _bs_values(
                        spot, strike, tau, sigma, rate, option_type
                    )
                    spread = max(0.05, 0.01 * price)
                    bid = max(0.01, price - 0.5 * spread)
                    ask = price + 0.5 * spread
                    rows.append(
                        [
                            "^SPX",
                            quote_date,
                            "SPXW",
                            expiration,
                            f"{strike:.3f}",
                            option_type,
                            "0.0000",
                            "0.0000",
                            "0.0000",
                            "0.0000",
                            "100",
                            "10",
                            f"{bid:.4f}",
                            "10",
                            f"{ask:.4f}",
                            f"{spot - 0.1:.4f}",
                            f"{spot + 0.1:.4f}",
                            f"{spot:.4f}",
                            f"{spot:.4f}",
                            f"{sigma:.4f}",
                            f"{delta:.4f}",
                            f"{gamma:.6f}",
                            "0.0000",
                            f"{vega:.4f}",
                            "0.0000",
                            "10",
                            f"{bid:.4f}",
                            "10",
                            f"{ask:.4f}",
                            f"{spot - 0.1:.4f}",
                            f"{spot + 0.1:.4f}",
                            f"{price:.4f}",
                            "1000",
                            "",
                        ]
                    )
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
        zip_path = root / f"UnderlyingOptionsEODCalcs_{quote_date}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr(f"UnderlyingOptionsEODCalcs_{quote_date}.csv", buffer.getvalue())
    return [str(item.astype("datetime64[D]")) for item in dates]
