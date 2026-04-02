from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from ivs_forecast.pipeline.run_experiment import write_summary_report


def _write_minimal_report_inputs(run_dir: Path) -> None:
    (run_dir / "manifests").mkdir(parents=True)
    pl.DataFrame(
        {
            "model_name": ["state_last", "state_var1", "ssvi_tcn_direct"],
            "rmse_iv": [0.1, 0.2, 0.3],
            "vega_rmse_iv": [0.11, 0.22, 0.33],
            "mae_iv": [0.01, 0.02, 0.03],
        }
    ).write_parquet(run_dir / "loss_panel.parquet")
    pl.DataFrame(
        {
            "model_name": ["state_last", "state_var1", "ssvi_tcn_direct"],
            "calendar_violation_count": [0.0, 0.0, 0.0],
            "butterfly_violation_count": [0.0, 0.0, 0.0],
        }
    ).write_parquet(run_dir / "forecast_ssvi_certification.parquet")
    pl.DataFrame(
        {
            "model_name": ["state_last", "state_var1", "ssvi_tcn_direct"],
            "price_rmse": [1.0, 2.0, 3.0],
            "inside_spread_rate": [0.5, 0.6, 0.7],
        }
    ).write_parquet(run_dir / "pricing_utility.parquet")
    pl.DataFrame(
        {
            "model_name": ["state_last", "state_var1", "ssvi_tcn_direct"],
            "bucket": ["all", "all", "all"],
            "hedged_pnl_rmse": [1.0, 2.0, 3.0],
            "hedged_pnl_mae": [0.5, 1.5, 2.5],
        }
    ).write_parquet(run_dir / "hedged_pnl_utility.parquet")
    pl.DataFrame(
        {
            "model_name": [
                "state_last",
                "state_last",
                "state_var1",
                "state_var1",
            ],
            "anchor_days": [30, 30, 30, 30],
            "net_return": [0.1, -0.1, 0.3, 0.1],
        }
    ).write_parquet(run_dir / "straddle_signal_utility.parquet")
    pl.DataFrame({"exclusion_reason": ["missing_history_window"]}).write_parquet(
        run_dir / "feature_row_exclusions.parquet"
    )
    (run_dir / "selected_model_configs.json").write_text(
        json.dumps(
            {
                "state_last": None,
                "state_var1": None,
                "ssvi_tcn_direct": {
                    "batch_size": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifests" / "run_resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "study": {
                    "underlying_symbol": "^SPX",
                    "option_root": "SPX",
                },
                "settlement": {
                    "proxy_time_eastern": "09:30:00",
                    "exact_clock": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_write_summary_report_rejects_incomplete_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "incomplete_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Run directory is incomplete"):
        write_summary_report(run_dir)


def test_write_summary_report_keeps_straddle_metrics_per_model(tmp_path) -> None:
    run_dir = tmp_path / "complete_run"
    run_dir.mkdir()
    _write_minimal_report_inputs(run_dir)

    summary_path = write_summary_report(run_dir)

    summary = summary_path.read_text(encoding="utf-8")
    assert "## Straddle Utility" in summary
    assert "- `state_last` / `30d`: mean_net_return=0.000000, hit_rate=0.5000, sharpe=0.0000" in summary
    assert "- `state_var1` / `30d`: mean_net_return=0.200000, hit_rate=1.0000, sharpe=1.4142" in summary
    assert "- `30d`: mean_net_return=" not in summary
