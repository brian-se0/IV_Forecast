from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset

from ivs_forecast.config import AppConfig
from ivs_forecast.data.clean import clean_contracts_day
from ivs_forecast.data.collapse_nodes import build_surface_nodes
from ivs_forecast.data.dfw import default_grid_definition, fit_dfw_surface, sample_dfw
from ivs_forecast.data.discovery import (
    audit_vendor_corpus,
    inventory_raw_files,
    parse_trade_date_from_filename,
)
from ivs_forecast.data.parity_forward import estimate_forward_terms
from ivs_forecast.data.schema import reconcile_schema


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "paths": {"raw_data_root": "D:/Options Data", "artifact_root": "./artifacts"},
            "study": {
                "underlying_symbol": "^SPX",
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "forecast_horizon_days": 1,
            },
        }
    )


def test_parse_trade_date_from_filename() -> None:
    assert parse_trade_date_from_filename(Path("UnderlyingOptionsEODCalcs_2020-01-02.zip")) == date(
        2020, 1, 2
    )


def test_reconcile_schema_accepts_observed_1545() -> None:
    observed = [
        "underlying_symbol",
        "quote_date",
        "root",
        "expiration",
        "strike",
        "option_type",
        "bid_1545",
        "ask_1545",
        "active_underlying_price_1545",
        "implied_volatility_1545",
        "delta_1545",
        "vega_1545",
        "trade_volume",
        "open_interest",
    ]
    reconciliation = reconcile_schema(observed, {name: "Float64" for name in observed})
    assert reconciliation.pass_status is True
    assert reconciliation.missing_required_columns == []


def test_clean_forward_collapse_and_dfw() -> None:
    quote_date = date(2020, 1, 2)
    expiration = date(2020, 3, 2)
    rows = []
    for strike, sigma in [(90.0, 0.22), (100.0, 0.20), (110.0, 0.21)]:
        call_mid = max(0.0, 100.0 - strike) + 5.0
        put_mid = call_mid - (100.0 - strike)
        for option_type, mid, delta in [("C", call_mid, 0.5), ("P", put_mid, -0.5)]:
            rows.append(
                {
                    "underlying_symbol": "^SPX",
                    "quote_date": quote_date,
                    "root": "SPXW",
                    "expiration": expiration,
                    "strike": strike,
                    "option_type": option_type,
                    "bid_1545": mid - 0.1,
                    "ask_1545": mid + 0.1,
                    "active_underlying_price_1545": 100.0,
                    "implied_volatility_1545": sigma,
                    "delta_1545": delta,
                    "vega_1545": 1.0,
                    "trade_volume": 10,
                    "open_interest": 100,
                }
            )
    clean, _ = clean_contracts_day(pl.DataFrame(rows), _config())
    forward_terms, _ = estimate_forward_terms(clean)
    assert forward_terms.height == 1
    nodes, quality = build_surface_nodes(clean, forward_terms, _config())
    assert quality["modeling_valid"][0] is False or nodes.height >= 3
    fit = fit_dfw_surface(
        pl.DataFrame(
            {
                "quote_date": [quote_date] * 8,
                "m": np.array([-0.25, -0.15, -0.05, 0.0, 0.05, 0.12, 0.18, 0.24]),
                "tau": np.array([20, 35, 50, 65, 90, 120, 180, 240]) / 365.0,
                "node_iv": np.array([0.24, 0.22, 0.205, 0.20, 0.202, 0.208, 0.215, 0.225]),
                "node_vega": np.ones(8),
            }
        )
    )
    sampled = sample_dfw(fit.coefficients, default_grid_definition())
    assert sampled.shape == (154,)
    assert sampled.min() >= 0.01


def test_audit_vendor_corpus_succeeds_on_synthetic_data(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_synthetic_vendor_dataset(raw_root, n_dates=3)
    records = inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
    report = audit_vendor_corpus(
        records=records,
        underlying_symbol="^SPX",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 1, 31),
    )
    assert report["pass_status"] is True
    assert report["raw_zip_count"] == 3
    assert report["missing_required_columns"] == []
    assert report["selected_underlying_caveats"]["active_underlying_price_1545_usable"] is True


def test_audit_vendor_corpus_rejects_quote_date_mismatch(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_synthetic_vendor_dataset(raw_root, n_dates=1)
    zip_path = raw_root / "UnderlyingOptionsEODCalcs_2020-01-02.zip"
    with zipfile.ZipFile(zip_path) as handle:
        csv_name = handle.namelist()[0]
        payload = handle.read(csv_name).decode("utf-8")
    broken_payload = payload.replace("2020-01-02", "2020-01-03", 1)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(csv_name, broken_payload)
    records = inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
    with pytest.raises(ValueError, match="quote_date contents disagreed"):
        audit_vendor_corpus(
            records=records,
            underlying_symbol="^SPX",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 31),
        )


def test_audit_vendor_corpus_rejects_missing_calcs_columns(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_synthetic_vendor_dataset(raw_root, n_dates=1)
    zip_path = raw_root / "UnderlyingOptionsEODCalcs_2020-01-02.zip"
    with zipfile.ZipFile(zip_path) as handle:
        csv_name = handle.namelist()[0]
        frame = pl.read_csv(io.BytesIO(handle.read(csv_name)))
    frame = frame.drop("vega_1545")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(csv_name, frame.write_csv())
    records = inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
    with pytest.raises(ValueError, match="Calcs-required 15:45 fields were missing"):
        audit_vendor_corpus(
            records=records,
            underlying_symbol="^SPX",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 31),
        )
