# IVS Forecast

A reproducible framework for **next-day implied-volatility-surface forecasting, model comparison, and downstream utility evaluation** using Cboe Option EOD Summary files.

## What this repository does

The repository takes daily vendor ZIP files, builds a standardized implied-volatility-surface panel, forecasts the next trading day’s 15:45 surface, and evaluates those forecasts in two ways:

1. **Statistical forecast accuracy**
   - contract/node-level IV errors
   - vega-weighted losses
   - Diebold-Mariano tests
   - Model Confidence Set analysis

2. **Downstream practical usefulness**
   - mark-to-mid pricing utility
   - delta-hedged marking-error utility
   - ATM straddle directional utility

This is not just a training repo. It is a **protocolized research system** with fixed stages, fixed artifacts, explicit runtime contracts, and strong chronology controls.

## Methodological summary

The design follows a literature-grounded two-step logic.

First, irregular option observations on each date are transformed into a **fixed sampled surface** on a standard grid. The grid follows the 154-point sampled representation used by Zhang, Li, and Zhang (2023), who show that sampled surfaces are among the strongest feature representations for IVS prediction.

Second, models forecast the next-day sampled surface. A shared no-arbitrage-aware reconstruction network then maps the sampled forecast back to arbitrary query points for evaluation on the realized next-day option chain. This follows the core idea of the arbitrage-aware Step-2 reconstruction used in Zhang, Li, and Zhang (2023), while the broader construction/smoothing rationale is supported by Cont and Da Fonseca (2002), Gatheral and Jacquier (2014), Ackerer, Tagasovska, and Vatter (2020), and later smoothing work such as Wiedemann, Jacquier, and Gonon (2025).

Chronology safety is treated as part of the method. All splits are ordered, all tuning is walk-forward, and no future data are allowed into scalers, PCA loadings, hyperparameter selection, or reconstructor training. This is especially important in light of recent IV-forecasting leakage critiques and the broader time-series evaluation literature.

## Supported scientific scope

Version 1 is intentionally scoped to:

- **one underlying per experiment**
- **index underlyings only**
- default validated symbol: **`^SPX`**
- **one homogeneous option root per modeling date**
- **1-day-ahead forecasting**
- target timestamp: **next-day 15:45 ET**

The raw vendor product covers stocks, ETFs, and indices, but the validated methodology path in this repo is index-first. Extending the same design to equity/ETF options is future work.

## Raw data expectations

The raw data are **not stored in the repository**.

The code expects external daily ZIP files named:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

The user’s local dataset is expected to reside under a configurable root such as:

- `D:\Options Data`

The attached vendor layout documentation shows that the product contains:

- a 15:45 ET options snapshot,
- an end-of-day summary,
- vendor-calculated 15:45 implied volatility and Greeks when the **calcs** product is ordered,
- daily OHLC / volume / VWAP / open interest fields.

This repo uses:

- **15:45 vendor IV and Greeks** for surface construction,
- same-day summary/activity fields as end-of-day features,
- next-day 15:45 data as the forecast target.

The canonical observed schema in this repository is the vendor/local `*_1545` field family, for example:

- `active_underlying_price_1545`
- `implied_volatility_1545`
- `delta_1545`
- `vega_1545`

If stale documentation refers to `*_15453`, the pipeline records that in the reconciliation report rather than treating those names as required columns.

### Important vendor notes

- The `1545` columns remain named `1545` on early-close days.
- `implied_volatility_1545` may be zero when the vendor calc engine cannot compute a valid IV; those rows are treated as invalid observations for IV-surface modeling.
- Some underlying quote fields may be zero, especially for indices.
- `root` is treated as the option-class key. The v1 sampled-surface path fails fast on mixed-root dates instead of blending classes such as `SPX` and `SPXW` into one surface.
- This repo requires **calcs-included** files and will fail on quote-only files.

### Documentation-vs-local-data verification

The current vendor web page and the local dataset may not agree perfectly on historical coverage. The implementation therefore:

- inventories the **actual observed local files**,
- compares observed columns to vendor documentation,
- writes a reconciliation report, and
- uses the validated local inventory rather than assuming the current marketing-page coverage window.

## How the pipeline works

1. **Inventory and validate raw ZIPs**
2. **Ingest and clean the selected underlying’s option chain**
3. **Estimate per-expiry forward prices from matched call-put quotes**
4. **Collapse calls/puts into unique surface nodes**
5. **Fit the Dumas-Fleming-Whaley interpolator and sample a fixed 154-point surface**
6. **Build date-level features and next-day targets**
7. **Train a shared reconstruction network**
8. **Train and compare a fixed model set**
9. **Evaluate statistical accuracy and downstream utility**
10. **Write immutable artifacts and summary reports**

## Supported model set

The benchmark set is deliberately small.

### `rw_last`
No-change baseline on the sampled surface.

### `pca_var1`
Three-factor PCA surface model with a VAR(1) forecast on factor scores.

### `xgb_direct`
GPU XGBoost model that forecasts the sampled surface directly from sampled-surface summaries and scalar regime features.

### `lstm_direct`
GPU LSTM model that forecasts the full sampled surface jointly from 22-day, 5-day, and current state summaries.

## Repository structure

```text
configs/                 experiment and model configs
docs/                    methodology, vendor, and artifact-contract notes
src/ivs_forecast/        implementation package
tests/                   unit and integration tests
artifacts/               generated outputs (not source-controlled in full)
````

## Python and environment

This project standardizes on **Python 3.14.3**.

Use `uv`:

```bash
uv sync
```

The repository assumes a modern NVIDIA GPU for:

* reconstruction network training
* LSTM training
* XGBoost training

CPU-only fallback is intentionally not implemented for GPU-required stages.

## Configuration

Copy the local config template and point it at your external raw data:

```yaml
paths:
  raw_data_root: "D:/Options Data"
  artifact_root: "./artifacts"

study:
  underlying_symbol: "^SPX"
  start_date: "2004-01-02"
  end_date: "2021-04-09"
  forecast_horizon_days: 1
```

Windows note:

* use either forward slashes (`D:/Options Data`) or escaped backslashes.

## Canonical commands

### 1. Verify data and schema

```bash
uv run ivs-forecast verify-data --config configs/experiments/spx_1d.yaml
```

This checks:

* ZIP readability
* one-CSV-per-ZIP integrity
* filename-vs-`quote_date` consistency across the full corpus
* required and calcs-required columns
* documentation-vs-observed schema differences
* selected-underlying caveats and early-close audit metadata

### 2. Run the full experiment

```bash
uv run ivs-forecast run --config configs/experiments/spx_1d.yaml
```

This is the canonical end-to-end path.

For a shorter integration-style run that still uses the same core pipeline code, use:

```bash
uv run ivs-forecast run --config configs/experiments/spx_smoke.yaml
```

### 3. Generate the final summary report

```bash
uv run ivs-forecast report --run-dir artifacts/runs/<run_id>
```

## What gets written

A successful run writes:

* raw inventory and schema reconciliation reports
* human-readable data audit report
* partitioned cleaned-contract datasets plus a date index
* partitioned surface-node datasets plus a date index
* forward-estimation diagnostics
* fixed-grid sampled surfaces
* feature/target datasets
* split manifest
* reconstructor artifacts
* per-model forecast artifacts
* loss panels
* arbitrage diagnostics
* DM and MCS outputs
* pricing / hedging / straddle utility panels
* markdown summary report

## How to interpret outputs

### Statistical outputs

* `loss_panel.parquet` is the master date-by-model loss table.
* `dm_tests.json` tells you whether forecast differences are statistically meaningful.
* `mcs_results.json` identifies the superior-model set under the Model Confidence Set procedure.

### Arbitrage diagnostics

These do not claim dynamic no-arbitrage. They report how well the reconstruction layer respects static-arbitrage conditions on a dense penalty grid.

### Downstream utility outputs

* `pricing_utility.parquet` measures how well the forecasted surface marks next-day option mids.
* `hedged_pnl_utility.parquet` measures how well the forecasted next-day mark predicts delta-hedged one-day PnL.
* `straddle_signal_utility.parquet` measures stylized ATM-straddle utility from forecasted ATM IV changes.

## Reproducibility guarantees

The repository is built to make reruns and audit trails easy.

Every run records:

* resolved config
* git commit
* package versions
* Python version
* CUDA/device information
* per-model device assignments
* random seed
* upstream artifact lineage

Chronology safety is enforced by design:

* ordered train / validation / test blocks
* walk-forward tuning
* expanding-window test refits
* no shuffled splits
* no future-fitted scalers or PCA objects

## Hardware expectations

Recommended:

* NVIDIA GPU with current CUDA-compatible PyTorch and XGBoost installs
* enough local storage for parquet artifacts
* enough RAM to process one day’s chain comfortably

This repo is designed for large historical datasets, but it is not designed to brute-force everything in memory.

## Key assumptions and limitations

* v1 is validated on **index underlyings**, not all OPRA underlyings.
* v1 forecasts **next-day 15:45** surfaces, not intraday trajectories.
* static-arbitrage awareness is implemented in the reconstruction stage; dynamic arbitrage is out of scope.
* the repo does not forecast which contracts will exist tomorrow; it forecasts the surface and evaluates on the realized next-day support.
* no external rates/dividend curves are required; forward quantities are estimated from matched call-put quotes.
* downstream trading diagnostics are stylized research diagnostics, not production execution logic.
* quote-only vendor files are unsupported.
* mixed-root dates are unsupported in the single-surface v1 path and fail fast until a homogeneous root policy is supplied.

## Literature grounding

The design is driven mainly by:

* **Cont and Da Fonseca (2002)** for IVS factor structure,
* **Chalamandaris and Tsekrekos (2010)** and **Kearney, Shang, and Sheenan (2019)** for out-of-sample IVS forecasting and economic-value framing,
* **Zhang, Li, and Zhang (2023)** for the two-step sampled-surface + DNN-reconstruction forecasting logic,
* **Gatheral and Jacquier (2014)** and related arbitrage-free surface papers for construction constraints,
* **Ackerer, Tagasovska, and Vatter (2020)** and later neural smoothing papers for arbitrage-aware surface completion,
* **Hansen, Lunde, and Nason (2011)** for MCS,
* **Hull and White (2017)** for hedging utility motivation,
* **recent leakage-aware IV forecasting work** for chronology-safe evaluation discipline.

## Practical bottom line

This repository is for users who want a **reproducible, leakage-safe, artifact-heavy research system** for comparing next-day implied-volatility-surface forecasts under a common protocol.

It is intentionally strict:

* one supported data path,
* one supported methodology path,
* fixed models,
* fixed artifacts,
* explicit failures instead of silent degradation.
