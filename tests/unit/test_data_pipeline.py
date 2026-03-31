from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError
from tests.fixtures.synthetic_vendor import write_synthetic_vendor_dataset

from ivs_forecast.config import AppConfig
from ivs_forecast.data.clean import clean_contracts_day
from ivs_forecast.data.collapse_nodes import build_surface_nodes
from ivs_forecast.data.discovery import (
    audit_vendor_corpus,
    inventory_raw_files,
    parse_trade_date_from_filename,
    raw_corpus_contract,
)
from ivs_forecast.data.early_closes import load_early_close_calendar
from ivs_forecast.data.parity_forward import estimate_forward_terms
from ivs_forecast.data.schema import reconcile_schema
from ivs_forecast.data.time_to_settlement import (
    settlement_policy_record,
    settlement_timestamp_eastern,
    snapshot_timestamp_eastern,
    year_fraction_act365,
)
from ivs_forecast.pipeline.build_data import build_data_stage


def _config(raw_root: str = "D:/Options Data", artifact_root: str = "./artifacts") -> AppConfig:
    return AppConfig.model_validate(
        {
            "paths": {"raw_data_root": raw_root, "artifact_root": artifact_root},
            "study": {
                "underlying_symbol": "^SPX",
                "option_root": "SPX",
                "start_date": "2020-01-02",
                "end_date": "2020-12-31",
                "forecast_horizon_days": 1,
                "min_surface_nodes": 3,
                "min_valid_expiries": 1,
            },
            "split": {
                "validation_size": 2,
                "test_size": 2,
                "min_train_size": 3,
                "refit_frequency": 1,
            },
            "runtime": {"seed": 20260329, "overwrite": False, "run_id": "unit_run"},
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


def test_study_config_requires_option_root() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
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


def test_snapshot_and_settlement_timestamps_apply_spx_soq_proxy_rules() -> None:
    normal_snapshot = snapshot_timestamp_eastern(date(2020, 11, 27), is_early_close=False)
    early_snapshot = snapshot_timestamp_eastern(date(2020, 11, 27), is_early_close=True)
    config = _config()
    settlement = settlement_timestamp_eastern(date(2020, 12, 18), "SPX", config.settlement)
    policy_record = settlement_policy_record("SPX", config.settlement)
    assert normal_snapshot.hour == 15 and normal_snapshot.minute == 45
    assert early_snapshot.hour == 12 and early_snapshot.minute == 45
    assert settlement.hour == 9 and settlement.minute == 30
    assert policy_record["settlement_style"] == "AM_SOQ_PROXY"
    assert policy_record["proxy_time_eastern"] == "09:30"
    assert policy_record["exact_clock"] is False
    normal_tau = year_fraction_act365(normal_snapshot, settlement)
    early_tau = year_fraction_act365(early_snapshot, settlement)
    assert early_tau > normal_tau


def test_early_close_manifest_is_loaded_from_checked_in_resource() -> None:
    calendar = load_early_close_calendar()
    july_entries = calendar.entries_in_range(date(2021, 7, 1), date(2021, 7, 5))
    assert calendar.calendar_name == "cboe_us_options_early_closes_manifest_v1"
    assert july_entries[0].quote_date == date(2021, 7, 2)
    assert july_entries[0].market_close_time_eastern == "12:45"


def test_clean_forward_and_collapse_are_root_explicit() -> None:
    quote_date = date(2020, 1, 2)
    expiration = date(2020, 3, 20)
    rows = []
    for root_name, sigma_shift in [("SPX", 0.0), ("SPXW", 0.03)]:
        for strike, sigma in [(2900.0, 0.22 + sigma_shift), (3000.0, 0.20 + sigma_shift), (3100.0, 0.21 + sigma_shift)]:
            call_mid = max(0.0, 3000.0 - strike) + 40.0
            put_mid = call_mid - (3000.0 - strike)
            for option_type, mid, delta in [("C", call_mid, 0.5), ("P", put_mid, -0.5)]:
                rows.append(
                    {
                        "underlying_symbol": "^SPX",
                        "quote_date": quote_date,
                        "root": root_name,
                        "expiration": expiration,
                        "strike": strike,
                        "option_type": option_type,
                        "bid_1545": mid - 0.1,
                        "ask_1545": mid + 0.1,
                        "active_underlying_price_1545": 3000.0,
                        "implied_volatility_1545": sigma,
                        "delta_1545": delta,
                        "vega_1545": 1.0,
                        "trade_volume": 10,
                        "open_interest": 100,
                    }
                )
    clean, _ = clean_contracts_day(pl.DataFrame(rows), _config())
    assert clean["root"].unique().to_list() == ["SPX"]
    forward_terms, diagnostics = estimate_forward_terms(clean)
    assert forward_terms.height == 1
    assert {item.root for item in diagnostics} == {"SPX"}
    nodes, quality = build_surface_nodes(clean, forward_terms, _config())
    assert nodes["option_root"].unique().to_list() == ["SPX"]
    assert quality["root_count"][0] == 1


def test_audit_vendor_corpus_reports_root_coverage(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_synthetic_vendor_dataset(raw_root, n_dates=3, include_secondary_root=True)
    records = inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
    report = audit_vendor_corpus(
        records=records,
        underlying_symbol="^SPX",
        option_root="SPX",
        start_date=date(2020, 1, 2),
        end_date=date(2020, 1, 31),
    )
    assert report["pass_status"] is True
    assert report["selected_root_coverage"]["dates_with_option_root"] == 3
    assert report["selected_root_coverage"]["dates_missing_option_root"] == []
    assert report["selected_underlying_caveats"]["root_row_counts"]["SPX"] > 0


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
            option_root="SPX",
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
            option_root="SPX",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 31),
        )


def test_build_data_fails_when_configured_root_is_missing(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    artifact_root = tmp_path / "artifacts"
    write_synthetic_vendor_dataset(raw_root, n_dates=3, option_root="SPXW")
    config = _config(str(raw_root), str(artifact_root))
    with pytest.raises(ValueError, match="configured option_root was absent"):
        build_data_stage(config)


def test_inventory_raw_files_supports_nested_daily_layout(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    nested_root = raw_root / "year=2020" / "month=01"
    write_synthetic_vendor_dataset(nested_root, n_dates=3)
    records = inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
    assert len(records) == 3
    contract = raw_corpus_contract(raw_root, records, date(2020, 1, 2), date(2020, 1, 31), "SPX")
    assert contract["grouping_mode"] == "nested_daily"


def test_inventory_raw_files_rejects_grouped_archive_without_daily_zips(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    grouped_path = raw_root / "UnderlyingOptionsEODCalcs_2020-01.zip"
    with zipfile.ZipFile(grouped_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("notes.txt", "grouped archive placeholder")
    with pytest.raises(FileNotFoundError, match="grouped monthly/yearly calcs archives"):
        inventory_raw_files(raw_root, date(2020, 1, 2), date(2020, 1, 31))
