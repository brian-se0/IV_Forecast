from __future__ import annotations

from datetime import date

import polars as pl

from ivs_forecast.data.parity_forward import ForwardEstimationDiagnostics
from ivs_forecast.reporting.data_quality import (
    DailyBuildDiagnostics,
    build_benchmark_contract,
    build_forward_invalid_reasons_summary,
    build_stage_coverage_by_year,
    build_stage_loss_by_date,
)


def test_build_stage_loss_by_date_assigns_first_failed_stage_and_reason() -> None:
    d1 = date(2020, 1, 2)
    d2 = date(2020, 1, 3)
    d3 = date(2020, 1, 6)
    d4 = date(2020, 1, 7)
    d5 = date(2020, 1, 8)
    d6 = date(2020, 1, 9)
    d7 = date(2020, 1, 10)
    trading_index = pl.DataFrame(
        {
            "quote_date": [d1, d2, d3, d4, d5, d6, d7],
            "option_root": ["SPX"] * 7,
            "trading_day_index": list(range(7)),
            "next_trading_date": [d2, d3, d4, d5, d6, d7, None],
            "has_surface_state": [False, False, False, True, False, True, True],
            "surface_state_row_index": [None, None, None, 0, None, 1, 2],
        }
    )
    diagnostics = {
        d1: DailyBuildDiagnostics(quote_date=d1, option_root="SPX", subset_rows=120, clean_contract_rows=0),
        d2: DailyBuildDiagnostics(
            quote_date=d2,
            option_root="SPX",
            subset_rows=120,
            clean_contract_rows=60,
            forward_total_expiries=3,
            forward_valid_expiries=0,
            forward_invalid_expiries=3,
            forward_invalid_reason_codes=["fewer_than_3_matched_strikes"],
        ),
        d3: DailyBuildDiagnostics(
            quote_date=d3,
            option_root="SPX",
            subset_rows=120,
            clean_contract_rows=60,
            forward_term_rows=4,
            forward_total_expiries=4,
            forward_valid_expiries=4,
            surface_node_count=18,
            valid_expiry_count=3,
            root_count=1,
            modeling_valid=False,
        ),
        d4: DailyBuildDiagnostics(
            quote_date=d4,
            option_root="SPX",
            subset_rows=120,
            clean_contract_rows=60,
            forward_term_rows=5,
            forward_total_expiries=5,
            forward_valid_expiries=5,
            surface_node_count=55,
            valid_expiry_count=5,
            root_count=1,
            modeling_valid=True,
            has_surface_state=True,
        ),
        d5: DailyBuildDiagnostics(quote_date=d5, option_root="SPX", subset_rows=120, clean_contract_rows=0),
        d6: DailyBuildDiagnostics(
            quote_date=d6,
            option_root="SPX",
            subset_rows=120,
            clean_contract_rows=60,
            forward_term_rows=5,
            forward_total_expiries=5,
            forward_valid_expiries=5,
            surface_node_count=60,
            valid_expiry_count=5,
            root_count=1,
            modeling_valid=True,
            has_surface_state=True,
        ),
        d7: DailyBuildDiagnostics(
            quote_date=d7,
            option_root="SPX",
            subset_rows=120,
            clean_contract_rows=60,
            forward_term_rows=5,
            forward_total_expiries=5,
            forward_valid_expiries=5,
            surface_node_count=61,
            valid_expiry_count=5,
            root_count=1,
            modeling_valid=True,
            has_surface_state=True,
        ),
    }
    features_targets = pl.DataFrame(
        {
            "quote_date": [d6],
            "target_date": [d7],
            "option_root": ["SPX"],
        }
    )
    feature_exclusions = pl.DataFrame(
        {
            "quote_date": [d1, d2, d3, d4, d5],
            "target_date": [d2, d3, d4, d5, d6],
            "option_root": ["SPX"] * 5,
            "exclusion_reason": [
                "missing_origin_state",
                "missing_origin_state",
                "missing_origin_state",
                "missing_target_state",
                "missing_origin_state",
            ],
        }
    )

    stage_loss = build_stage_loss_by_date(
        trading_date_index=trading_index,
        daily_diagnostics=diagnostics,
        features_targets=features_targets,
        feature_exclusions=feature_exclusions,
        min_surface_nodes=40,
        min_valid_expiries=4,
    )

    rows = {row["quote_date"]: row for row in stage_loss.iter_rows(named=True)}
    assert rows[d1]["first_failed_stage"] == "cleaning"
    assert rows[d1]["first_failure_reason_codes"] == ["cleaning_eliminated_all_contracts"]
    assert rows[d2]["first_failed_stage"] == "forward_estimation"
    assert rows[d2]["first_failure_reason_codes"] == ["fewer_than_3_matched_strikes"]
    assert rows[d3]["first_failed_stage"] == "node_construction"
    assert rows[d3]["first_failure_reason_codes"] == [
        "below_min_surface_nodes",
        "below_min_valid_expiries",
    ]
    assert rows[d4]["first_failed_stage"] == "feature_target_formation"
    assert rows[d4]["first_failure_reason_codes"] == ["missing_target_state"]
    assert rows[d6]["feature_row_kept"] is True
    assert rows[d6]["first_failed_stage"] is None
    assert rows[d7]["has_next_trading_date"] is False
    assert rows[d7]["feature_row_kept"] is False

    coverage = build_stage_coverage_by_year(stage_loss)
    assert coverage == [
        {
            "year": 2020,
            "raw_trading_days": 7,
            "days_after_cleaning": 5,
            "days_with_forward_terms": 4,
            "days_with_node_quality": 4,
            "modeling_valid_days": 3,
            "ssvi_state_days": 3,
            "feature_candidate_days": 6,
            "feature_rows": 1,
            "terminal_days": 1,
            "forward_invalid_expiry_count": 3,
            "first_failed_stage_counts": {
                "cleaning": 2,
                "feature_target_formation": 1,
                "forward_estimation": 1,
                "node_construction": 1,
            },
            "first_failure_reason_counts": {
                "below_min_surface_nodes": 1,
                "below_min_valid_expiries": 1,
                "cleaning_eliminated_all_contracts": 2,
                "fewer_than_3_matched_strikes": 1,
                "missing_target_state": 1,
            },
        }
    ]


def test_build_forward_invalid_reasons_and_benchmark_contract_are_machine_readable() -> None:
    diagnostics = [
        ForwardEstimationDiagnostics(
            quote_date="2020-01-02",
            root="SPX",
            expiration="2020-02-14",
            matched_pairs_before_prune=2,
            matched_pairs_after_prune=0,
            invalid_reason="fewer_than_3_matched_strikes",
        ),
        ForwardEstimationDiagnostics(
            quote_date="2020-01-03",
            root="SPX",
            expiration="2020-02-21",
            matched_pairs_before_prune=4,
            matched_pairs_after_prune=2,
            invalid_reason="fewer_than_3_after_mad_prune",
        ),
        ForwardEstimationDiagnostics(
            quote_date="2020-01-03",
            root="SPX",
            expiration="2020-03-20",
            matched_pairs_before_prune=4,
            matched_pairs_after_prune=4,
            invalid_reason=None,
        ),
    ]
    forward_summary = build_forward_invalid_reasons_summary(diagnostics)
    assert forward_summary["invalid_expiry_count"] == 2
    assert forward_summary["overall_by_reason"] == {
        "fewer_than_3_after_mad_prune": 1,
        "fewer_than_3_matched_strikes": 1,
    }
    assert forward_summary["by_year"] == [
        {
            "year": 2020,
            "invalid_expiry_count": 2,
            "reason_counts": {
                "fewer_than_3_after_mad_prune": 1,
                "fewer_than_3_matched_strikes": 1,
            },
        }
    ]

    benchmark_contract = build_benchmark_contract(
        raw_corpus_contract={
            "window_coverage": {
                "requested_window": {"start_date": "2020-01-02", "end_date": "2020-01-10"},
                "observed_window": {"start_date": "2020-01-02", "end_date": "2020-01-10"},
                "matches_requested_window": True,
                "coverage_status": "exact",
            }
        },
        trading_date_index=pl.DataFrame(
            {
                "quote_date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                "next_trading_date": [date(2020, 1, 3), date(2020, 1, 6), None],
            }
        ),
        ssvi_state=pl.DataFrame(
            {
                "quote_date": [date(2020, 1, 3), date(2020, 1, 6)],
                "option_root": ["SPX", "SPX"],
            }
        ),
        features_targets=pl.DataFrame(
            {
                "quote_date": [date(2020, 1, 3)],
                "target_date": [date(2020, 1, 6)],
                "option_root": ["SPX"],
            }
        ),
        feature_exclusions=pl.DataFrame(
            {
                "quote_date": [date(2020, 1, 2)],
                "target_date": [date(2020, 1, 3)],
                "option_root": ["SPX"],
                "exclusion_reason": ["missing_history_window"],
            }
        ),
        minimum_history_days=22,
    )
    assert benchmark_contract["forecastable_window"]["history_window_days"] == 22
    assert benchmark_contract["forecastable_window"]["ssvi_state_window"] == {
        "start_date": "2020-01-03",
        "end_date": "2020-01-06",
        "row_count": 2,
    }
    assert benchmark_contract["forecastable_window"]["feature_origin_window"] == {
        "start_date": "2020-01-03",
        "end_date": "2020-01-03",
        "row_count": 1,
    }
    assert benchmark_contract["forecastable_window"]["feature_exclusion_counts"] == {
        "missing_history_window": 1
    }
