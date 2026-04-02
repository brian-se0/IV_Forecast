from __future__ import annotations

import pytest

from ivs_forecast.pipeline.run_experiment import write_summary_report


def test_write_summary_report_rejects_incomplete_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "incomplete_run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Run directory is incomplete"):
        write_summary_report(run_dir)
