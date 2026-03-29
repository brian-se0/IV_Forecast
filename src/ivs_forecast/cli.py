from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ivs_forecast.config import load_config
from ivs_forecast.logging_utils import configure_logging

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging.")) -> None:
    configure_logging(verbose=verbose)


@app.command("verify-data")
def verify_data(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    raw_data_root: Path | None = typer.Option(None, "--raw-data-root"),
) -> None:
    from ivs_forecast.pipeline.build_data import verify_data_stage

    resolved = load_config(config, raw_data_root_override=raw_data_root)
    outputs = verify_data_stage(resolved)
    console.print(f"Wrote inventory to {outputs['inventory_path']}")
    console.print(f"Wrote reconciliation to {outputs['schema_report_path']}")
    console.print(f"Wrote audit report to {outputs['audit_report_path']}")


@app.command("build-data")
def build_data(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    raw_data_root: Path | None = typer.Option(None, "--raw-data-root"),
) -> None:
    from ivs_forecast.pipeline.build_data import build_data_stage

    resolved = load_config(config, raw_data_root_override=raw_data_root)
    outputs = build_data_stage(resolved)
    console.print(f"Built data artifacts under {outputs['run_root']}")


@app.command("run")
def run_experiment(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    raw_data_root: Path | None = typer.Option(None, "--raw-data-root"),
) -> None:
    from ivs_forecast.pipeline.run_experiment import run_experiment as run_pipeline

    resolved = load_config(config, raw_data_root_override=raw_data_root)
    run_dir = run_pipeline(resolved)
    console.print(f"Run completed under {run_dir}")


@app.command("report")
def report(run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False)) -> None:
    from ivs_forecast.pipeline.run_experiment import write_summary_report

    summary_path = write_summary_report(run_dir)
    console.print(f"Wrote summary report to {summary_path}")
