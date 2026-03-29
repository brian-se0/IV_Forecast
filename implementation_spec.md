# Implementation specification

## 1. Status, authoritative inputs, and scope

This specification is the project-defining source of truth for the repository.

Authoritative inputs used to define this design:

1. The uploaded `lit_review.zip` archive, treated as the primary academic corpus.
2. The attached `Option_EOD_Summary_Layout.pdf`.
3. The current Cboe DataShop product page for Option EOD Summary, used only for vendor-product interpretation and to identify documentation-vs-observed-data reconciliation requirements.

Archive review note:

- The archive is readable.
- The actual archive contains **75 PDF papers**.
- The extracted `canonical_inventory.json` under `extracted/` reports **72** because three distinct papers were incorrectly collapsed by duplicate handling.
- For this project, the **actual archive contents are authoritative**, not the undercounting inventory.

The repository defined here is an **end-to-end, chronology-safe, reproducible framework** for:

- ingesting external Cboe daily option summary files,
- building standardized implied-volatility-surface artifacts over time,
- forecasting next-day surfaces,
- comparing a small, fixed model set under a common protocol, and
- evaluating both statistical forecast quality and downstream practical usefulness.

## 2. Project goals

### 2.1 Goals

The repository must:

- produce a **single supported research pipeline** from raw vendor ZIP files to final evaluation artifacts;
- enforce **chronology safety** and explicit leakage controls at every stage;
- create **fixed intermediate artifacts** with stable schemas and verification reports;
- compare a **small, justified model set** under a shared out-of-sample protocol;
- evaluate forecasts using both:
  - statistical accuracy metrics and formal forecast comparison tests, and
  - downstream diagnostics that approximate marking and hedging usefulness;
- make reproducibility part of the method, not a documentation afterthought.

### 2.2 Non-goals

Version 1 of the repository must **not** attempt to do the following:

- intraday forecasting;
- pooled multi-underlying training across heterogeneous assets;
- American-option early-exercise modeling;
- dynamic-arbitrage modeling;
- forecast combination or ensemble pooling;
- deep-RL hedging as a primary benchmark;
- scenario generation / simulation as a required pipeline stage;
- direct calibration of full stochastic-volatility models as the primary forecasting path;
- fallback coordinate systems, fallback model families, or fallback data sources.

## 3. Scientific scope

### 3.1 Supported research scope

The validated research scope for v1 is:

- **one underlying per experiment**;
- **index underlyings only** for the supported methodology path;
- default validated underlying: **`^SPX`**;
- forecast horizon: **1 next valid trading day**;
- target timestamp: **next-day 15:45 U.S. Eastern implied-vol surface**.

### 3.2 Why the scope is index-first

This is a deliberate methodological and engineering choice.

The literature base most directly relevant to this repository’s design is strongest on index and FX surfaces, especially for surface dynamics, factor structure, arbitrage-aware smoothing, and short-horizon forecasting (Cont and Da Fonseca, 2002; Chalamandaris and Tsekrekos, 2010; Zhang, Li, and Zhang, 2023). The vendor data product covers stocks, ETFs, and indices, but reliable forward-style surface coordinates and pricing diagnostics are materially easier and cleaner for index surfaces than for American-style equity/ETF options. Therefore:

- ingestion may catalog all underlyings;
- the **validated forecasting/evaluation protocol** in v1 is for index underlyings only;
- extending the same pipeline to equity/ETF options is deferred until the early-exercise and carry-treatment issues are explicitly redesigned.

This is a **project-level scoping decision**, not a claim that equities are unimportant.

## 4. Research questions and evaluation objectives

The repository must answer the following questions:

1. Can a standardized sampled-surface representation support reproducible next-day IV-surface forecasts that outperform a no-change baseline and a classical factor benchmark?
2. Under leakage-safe evaluation, do modern nonlinear forecasters materially improve out-of-sample IV-surface accuracy over classical baselines?
3. Does a shared no-arbitrage-aware surface reconstruction layer improve contract-level evaluation quality relative to working only on a sampled grid?
4. Do better statistical forecasts translate into better downstream marking and hedging-style diagnostics?

Evaluation is successful only if all models are compared on the **same dates**, **same forecast horizon**, and **same evaluation contract universe**.

## 5. Core methodological decisions

### 5.1 Surface coordinate system

Use:

- time to maturity in years: `tau = dte_days / 365.0`
- **log-forward moneyness**: `m = log(K / F_t(tau))`

This follows the literature preference for relative coordinates over raw strike, and aligns with the main forecasting design in Zhang, Li, and Zhang (2023), as well as surface-dynamics work such as Cont and Da Fonseca (2002).

### 5.2 Forward estimation

Because the vendor files do not include a complete term structure of rates and dividends, the repository must estimate a same-date, same-expiry forward from matched call-put quotes using put-call parity.

This is an **engineering assumption required by the data interface**, not a literature-settled methodological result.

### 5.3 Standardized state representation

The primary daily state is the **sampled surface (SAM)** on the fixed 154-point grid used by Zhang, Li, and Zhang (2023):

- moneyness ratios:
  - `0.60, 0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.20, 1.30, 1.50, 1.75, 2.00`
- maturities in days:
  - `10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730`

The grid is therefore `14 x 11 = 154` points.

Why SAM:

- Zhang, Li, and Zhang (2023) find that **sampling** and **VAE features** are the best-performing feature representations, while PCA is materially worse.
- SAM is chosen over VAE for v1 because it is more transparent, creates simpler artifact contracts, avoids latent-feature identifiability issues, and keeps the implementation aligned with a large historical data workflow.

This is a deliberate tie-break between two literature-supported strong options.

### 5.4 Daily standardization method

Use the Dumas-Fleming-Whaley polynomial interpolator (“DFW”) as the deterministic standardization layer:

`iv_hat(m, tau) = max(0.01, a0 + a1*m + a2*tau + a3*m^2 + a4*tau^2 + a5*m*tau)`

This is the same interpolation device used by Zhang, Li, and Zhang (2023) to sample a fixed surface from irregular observations.

Trade-off:

- large-scale construction work also shows strong performance from kernel smoothing approaches (e.g. OptionMetrics-style smoothing; Ulrich, Zimmer, and Merbecks, 2023);
- however, v1 adopts DFW because it matches the core two-step forecasting paper most closely and keeps the standardized-state contract simple and deterministic.

### 5.5 Reconstruction layer

Use a **shared no-arbitrage-aware PyTorch MLP reconstructor** that maps:

- sampled surface vector `F_t in R^154`
- query point `(m, tau)`

to `sigma_hat_t(m, tau)`.

The reconstructor must implement the same type of Step-2 idea as Zhang, Li, and Zhang (2023): a DNN that learns to reconstruct a continuous surface from sampled features while penalizing static-arbitrage violations. The positivity / differentiability / calendar / butterfly / tail conditions must be implemented from the **source PDF equations**, not from OCR or markdown extraction.

Important:

- the repository targets **static-arbitrage-aware** reconstruction;
- it does **not** claim to enforce dynamic no-arbitrage;
- if source markdown and source PDF disagree on an equation, the PDF is authoritative.

## 6. Forecast origin and chronology policy

### 6.1 Forecast origin

Forecasts are made **after the close of trade date `t`**, using all information available in file `t`, to predict the **15:45 implied-vol surface on the next valid trading date `t+1`**.

This choice is driven by vendor semantics:

- implied volatility and Greeks are available at 15:45;
- OHLC / VWAP / trade volume are daily summary fields;
- using close-of-day as the forecast origin allows same-day activity variables without leakage.

This is an engineering decision implied by the vendor file design.

### 6.2 Valid dates

The modeling calendar is the ordered sequence of dates for which all required upstream artifacts exist:

- readable raw ZIP;
- schema-valid CSV;
- sufficient cleaned quotes for the selected underlying;
- successful forward estimation for enough expiries;
- successful daily sampled-surface construction.

The repository must not assume that every business date in the range is valid.

## 7. Vendor data contract

### 7.1 Supported raw-file type

Supported raw files are only:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

Files corresponding to quote-only orders (`UnderlyingOptionsEODQuotes_*.zip`) are not supported for the forecasting pipeline because v1 requires vendor-supplied 15:45 implied volatilities and Greeks.

### 7.2 Expected ZIP contract

For each supported raw file:

- exactly one CSV member inside the ZIP;
- filename date must match the `quote_date` content;
- unreadable ZIPs, multiple CSV members, or missing CSV members are fatal errors.

### 7.3 Required columns

The following columns are required for v1:

- `underlying_symbol`
- `quote_date`
- `root`
- `expiration`
- `strike`
- `option_type`
- `bid_1545`
- `ask_1545`
- `active_underlying_price_15453`
- `implied_volatility_15453`
- `delta_15453`
- `vega_15453`
- `trade_volume`
- `open_interest`

Optional but ingested if present:

- `gamma_15453`
- `theta_15453`
- `rho_15453`
- `bid_eod`
- `ask_eod`
- `vwap`
- `open`, `high`, `low`, `close`

Ignored for modeling:

- `implied_underlying_price_15453` (deprecated)
- `delivery_code` (deprecated / empty)

### 7.4 Vendor caveats that must be respected

The code and docs must explicitly account for the following documented vendor behaviors:

- 15:45 fields stay named `1545` even on early-close days;
- some underlying bid/ask quote fields may be zero, especially for indices;
- implied volatility may be zero when the vendor calc engine lacks sufficient inputs or the option is below intrinsic value.

Zero implied volatility rows must be treated as **invalid observations for IV-surface modeling**, not as genuine zero-volatility points.

## 8. Documentation-vs-observed-data reconciliation rules

The implementation must produce a machine-readable reconciliation report comparing:

- documented vendor schema and caveats;
- observed local files;
- configured experiment assumptions.

Continue only when differences are non-material, such as:

- extra columns,
- column order changes,
- harmless type widening.

Stop with a clear error when differences are material, such as:

- missing required calc columns,
- systematic absence of 15:45 IV / Greeks,
- incompatible naming pattern,
- corrupted date semantics,
- repeated mismatches between filename date and file content.

The report must also note the historical-coverage discrepancy:

- the current vendor web page states one coverage window;
- the user’s local dataset may contain a different historical window;
- the repository must trust the **observed local inventory** after validation, not the marketing-page coverage claim.

## 9. Repository layout

```text
.
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ uv.lock
├─ configs/
│  ├─ local.example.yaml
│  ├─ experiments/
│  │  ├─ spx_1d.yaml
│  │  └─ spx_smoke.yaml
│  └─ models/
│     ├─ reconstructor.yaml
│     ├─ xgboost.yaml
│     └─ lstm.yaml
├─ docs/
│  ├─ methodology.md
│  ├─ vendor_dataset_contract.md
│  └─ artifact_contracts.md
├─ src/ivs_forecast/
│  ├─ cli.py
│  ├─ config.py
│  ├─ logging_utils.py
│  ├─ artifacts/
│  │  ├─ manifests.py
│  │  └─ hashing.py
│  ├─ data/
│  │  ├─ discovery.py
│  │  ├─ schema.py
│  │  ├─ ingest.py
│  │  ├─ clean.py
│  │  ├─ parity_forward.py
│  │  ├─ collapse_nodes.py
│  │  ├─ dfw.py
│  │  └─ sampled_surface.py
│  ├─ features/
│  │  ├─ windows.py
│  │  ├─ scalars.py
│  │  └─ dataset.py
│  ├─ models/
│  │  ├─ base.py
│  │  ├─ rw_last.py
│  │  ├─ pca_var1.py
│  │  ├─ xgb_direct.py
│  │  ├─ lstm_direct.py
│  │  └─ reconstructor.py
│  ├─ evaluation/
│  │  ├─ metrics.py
│  │  ├─ dm.py
│  │  ├─ mcs.py
│  │  ├─ arbitrage.py
│  │  ├─ pricing_mark.py
│  │  ├─ hedged_pnl.py
│  │  └─ straddle_signal.py
│  └─ pipeline/
│     ├─ build_data.py
│     ├─ train_models.py
│     ├─ forecast.py
│     └─ run_experiment.py
└─ tests/
   ├─ unit/
   ├─ integration/
   └─ fixtures/
````

## 10. Environment, dependencies, and Python version policy

### 10.1 Python version

Use **Python 3.14.3** as the project standard.

This is a deliberate project-level decision. It must be reflected consistently in:

* `pyproject.toml`
* `uv.lock`
* `README.md`
* `AGENTS.md`
* CI and local run instructions

### 10.2 Package manager

Use **`uv`** for environment creation and lockfile management.

### 10.3 Required dependencies

Use this stack unless the spec is formally revised:

* `polars`
* `pyarrow`
* `numpy`
* `scipy`
* `pydantic`
* `pyyaml`
* `typer`
* `rich`
* `torch`
* `xgboost`
* `pytest`
* `ruff`

Do **not** introduce parallel dataframe stacks, duplicate modeling stacks, or alternative gradient-boosting frameworks in v1.

### 10.4 Precision policy

* use `float64` for:

  * forward estimation,
  * DFW fits,
  * arbitrage diagnostics,
  * evaluation metrics;
* use `float32` for:

  * model tensors,
  * XGBoost inputs,
  * reconstructor/LSTM training.

## 11. External raw-data-root policy

Raw vendor data live outside the repository.

### 11.1 Configuration

The repository must not hardcode machine-specific paths.

Use config precedence:

1. CLI override
2. environment variable
3. YAML config

Canonical config fields:

```yaml
paths:
  raw_data_root: "D:/Options Data"
  artifact_root: "./artifacts"
study:
  underlying_symbol: "^SPX"
  start_date: "2004-01-02"
  end_date: "2021-04-09"
```

### 11.2 Path rules

* accept Windows paths;
* normalize internally with `pathlib`;
* do not copy raw ZIPs into the repo;
* do not mutate raw ZIPs.

## 12. Ingestion and raw validation pipeline

The pipeline stages are fixed and must run in this order.

### Stage 0: inventory

Create `raw_inventory.parquet` and `raw_inventory.json` containing, at minimum:

* file path
* trade date parsed from filename
* file size
* SHA256
* CSV member name
* readable/unreadable status

### Stage 1: header/schema validation

Read only the header and a small row sample first.

Write `vendor_schema_reconciliation.json` containing:

* documented columns
* observed columns
* missing required columns
* extra columns
* dtype inferences
* pass/fail status

### Stage 2: typed ingest

Stream each ZIP member directly into typed parquet without bulk-unzipping the whole history.

No in-memory all-history dataframe is allowed.

## 13. Contract cleaning and node construction

### 13.1 Raw contract filters

For the selected underlying, keep rows satisfying:

* `option_type in {C, P}`
* `strike > 0`
* `quote_date < expiration`
* `10 <= dte_days <= 730`
* `bid_1545 > 0`
* `ask_1545 > 0`
* `ask_1545 >= bid_1545`
* `implied_volatility_15453 > 0`
* `vega_15453 > 0`
* `active_underlying_price_15453 > 0`

Compute:

* `mid_1545 = 0.5 * (bid_1545 + ask_1545)`
* `spread_1545 = ask_1545 - bid_1545`
* `rel_spread_1545 = spread_1545 / max(mid_1545, 1e-6)`
* `tau = dte_days / 365.0`

### 13.2 Duplicate handling

The unique contract key is:

`(quote_date, underlying_symbol, root, expiration, strike, option_type)`

* exact duplicates with identical values may be deduplicated with a report;
* conflicting duplicates are fatal.

### 13.3 Forward estimation

For each `(quote_date, expiration)`:

1. match calls and puts by strike;
2. compute `y_j = C_mid_j - P_mid_j`;
3. fit weighted linear model `y_j = alpha + beta * K_j` with weights `min(vega_call_j, vega_put_j)`;
4. perform one outlier-pruning pass using a `5 x MAD` residual rule;
5. refit.

Infer:

* `discount_factor = -beta`
* `forward_price = alpha / discount_factor`

Invalidate the expiry if:

* fewer than 3 matched strikes remain,
* `discount_factor <= 0`,
* `forward_price <= 0`,
* fit returns NaN or inf.

### 13.4 Surface-node collapse

The forecasting target is a **single IV surface**, not separate call and put surfaces.

After forward estimation:

* transform each contract to `(m, tau)` with `m = log(strike / forward_price)`;
* for strike-expiry pairs with both call and put surviving, collapse to one node via vega-weighted IV average;
* if only one side survives, keep that node.

The resulting object is a set of unique surface nodes per date.

### 13.5 Minimum daily validity

A date is valid for modeling only if, after cleaning and node construction, it has:

* at least 40 unique surface nodes, and
* at least 4 valid expiries.

Otherwise the date is excluded from the modeling panel and the exclusion is recorded.

## 14. Standardized sampled surface

### 14.1 DFW fit

For each valid date, fit the DFW cross-sectional polynomial to the unique surface nodes using vega weights.

If the linear system is singular or numerically unstable, the date is invalid and must be excluded.

### 14.2 Fixed grid

Use the fixed 154-point grid from Section 5.3 exactly.

### 14.3 Stored surface forms

For every valid date store:

* sampled IV grid `iv_g000 ... iv_g153`
* sampled log-IV grid `logiv_g000 ... logiv_g153`
* grid definition file mapping `gNNN -> (m, tau)`

The sampled surface artifact is the canonical daily state used by all forecasting models.

## 15. Feature engineering and target definition

### 15.1 Primary predictors

For each valid date `t`, create:

* current sampled log-IV surface: `x_curr_t`
* 5-day average sampled log-IV surface: `x_ma5_t`
* 22-day average sampled log-IV surface: `x_ma22_t`

These are directly inspired by the weekly / monthly / current summaries used in Zhang, Li, and Zhang (2023), and the HAR-style logic of Corsi (2009).

### 15.2 Scalar regime features

Add:

* `underlying_logret_1`
* `underlying_logret_5`
* `underlying_logret_22`
* `log1p_total_trade_volume`
* `log1p_total_open_interest`
* `median_rel_spread_1545`

All scalar features are computed from information available by the close of date `t`.

### 15.3 Target

Default target:

* next valid date sampled log-IV surface `y_{t+1}`

The model does **not** forecast future chain composition. Contract-level evaluation is performed on the realized next-day contract universe, following the same logic as Zhang, Li, and Zhang (2023), who evaluate on the observed next-day support.

## 16. Supported model set

The supported models are fixed.

### 16.1 Baseline: `rw_last`

Forecast next-day sampled log-IV surface as the current-day sampled log-IV surface.

Rationale:

* a no-change benchmark is hard to beat in volatility forecasting;
* it is standard in the IVS literature to compare against random-walk / no-change baselines.

### 16.2 Classical benchmark: `pca_var1`

Pipeline:

1. fit PCA on training sampled log-IV surfaces;
2. retain **3 factors**;
3. fit a **VAR(1)** with intercept on factor scores;
4. forecast next-day score vector;
5. reconstruct next-day sampled log-IV surface.

Rationale:

* level / skew / convexity factor structure is foundational in Cont and Da Fonseca (2002);
* three-factor structure is the natural classical benchmark;
* a joint VAR(1) captures factor dependence more appropriately than fully separate univariate score models.

### 16.3 Nonlinear tree model: `xgb_direct`

Forecast each sampled-grid point directly from:

* `x_curr_t`
* `x_ma5_t`
* `x_ma22_t`
* scalar regime features

using one GPU XGBoost regressor per grid point.

Rationale:

* regression-tree approaches are established in IVS forecasting (Audrino and Colangelo, 2010);
* tree-based forecasting of time-varying IVS coefficients is directly relevant (Mathematics 2024 B-spline/tree paper);
* the leakage-review paper in the corpus shows that properly validated tree models can outperform more complex neural baselines when chronology is respected.

### 16.4 Sequence model: `lstm_direct`

Forecast the full sampled surface jointly with a multi-output LSTM.

Input sequence length is fixed at 3:

1. 22-day surface/scalar summary
2. 5-day surface/scalar summary
3. current-day surface/scalar state

Output is the full 154-dimensional next-day sampled log-IV vector.

Rationale:

* directly aligned with Zhang, Li, and Zhang (2023);
* consistent with later deep IV-forecasting papers such as Medvedev and Wang (2022) and related IV-forecasting work in the corpus.

### 16.5 Explicitly unsupported v1 model families

Do not implement in v1:

* VAE forecasting model as a required benchmark;
* transformers;
* ConvLSTM;
* neural operators for forecasting;
* model averaging / forecast pooling;
* deep-RL hedgers.

These remain extension topics.

## 17. Reconstructor design

### 17.1 Architecture

Use a shared MLP:

* input dimension: `154 + 2`
* hidden widths: `256, 256, 128`
* hidden activation: `SiLU`
* output activation: `Softplus`

### 17.2 Training target

Train on historical pairs:

* input: `(sampled_surface_t, m_i, tau_i)`
* target: observed node IV at `(m_i, tau_i)` on date `t`

### 17.3 Arbitrage penalties

Implement the static-arbitrage-aware penalty terms from **Section 3.4 / Proposition 1** of Zhang, Li, and Zhang (2023), copied exactly from the source PDF.

Penalty-grid policy:

* dense `m` grid, cubic-spaced and denser near ATM;
* dense `tau` grid, log-spaced between `10/365` and `730/365`;
* additional extreme-`m` tail points for the tail condition.

### 17.4 Reconstructor training window

The reconstructor is part of the chronology-safe pipeline.

It must be trained only on the currently available expanding window and re-fit on the same 21-day cadence as the predictive models. It is shared across models within a refit cycle.

## 18. Chronology-safe split, tuning, and test protocol

### 18.1 Outer split

Let valid dates be ordered as `D_1 < ... < D_N`.

Use:

* **test block**: final 504 valid dates
* **validation block**: preceding 252 valid dates
* **training block**: all earlier valid dates

Require at least 756 training dates after preprocessing. If this condition fails, stop.

### 18.2 Walk-forward tuning protocol

Hyperparameter tuning must use **walk-forward validation**, not random search with shuffled validation.

For each candidate hyperparameter set:

1. fit on the training block;
2. forecast the validation block in 21-day refit chunks;
3. within each chunk, produce one-step-ahead forecasts using realized history only;
4. aggregate validation losses on the same daily loss panel used later in test evaluation.

Primary tuning criterion:

1. lowest mean validation `rmse_iv`
2. tie-breaker: lowest mean validation `vega_rmse_iv`
3. models with materially worse arbitrage diagnostics than competing settings are rejected even if RMSE is slightly smaller

### 18.3 Final test protocol

After selecting hyperparameters:

1. merge training + validation blocks;
2. run the test block with the same 21-day expanding-window refit cadence;
3. train the reconstructor and all model families only on data available up to the current refit date.

No information from the test block may influence:

* scaling,
* PCA loadings,
* model hyperparameters,
* reconstructor weights.

## 19. Hyperparameter policy

### 19.1 Fixed choices

No tuning:

* `rw_last`
* `pca_var1` factor count = 3

### 19.2 XGBoost search space

Search only within:

* `max_depth`: `{4, 6}`
* `learning_rate`: `{0.03, 0.05}`
* `n_estimators`: `{300, 600}`
* `subsample`: `{0.8}`
* `colsample_bytree`: `{0.8}`
* `reg_lambda`: `{1.0, 5.0}`
* `min_child_weight`: `{1.0, 5.0}`

Use:

* `device = "cuda"`
* `tree_method = "hist"`

### 19.3 LSTM search space

Search only within:

* `hidden_size`: `{64, 128}`
* `dropout`: `{0.0, 0.1}`
* `learning_rate`: `{1e-3, 3e-4}`
* `batch_size`: `{128, 256}`

Fixed:

* layers = 2
* output dimension = 154
* optimizer = AdamW
* max epochs = 100
* early stopping patience = 10

## 20. Forecasting workflow

For each forecast date `t -> t+1`:

1. retrieve realized inputs available by the close of date `t`;
2. build model-family-specific feature representation;
3. forecast next-day sampled log-IV vector;
4. exponentiate to sampled IV vector;
5. reconstruct continuous surface through shared reconstructor;
6. query the reconstructed surface at the realized next-day node set and contract set;
7. write per-model daily prediction artifacts before computing evaluation summaries.

All models must use the same upstream sampled-surface artifact and the same downstream reconstructor within a refit window.

## 21. Evaluation metrics and formal comparison tests

### 21.1 Statistical forecast metrics

Compute on the realized next-day **unique surface nodes**:

* `rmse_iv`
* `mae_iv`
* `mape_iv_clipped`, with denominator `max(actual_iv, 1e-4)`
* `vega_rmse_iv`, vega-weighted RMSE using normalized node vegas

Primary reported metric:

* `rmse_iv`

### 21.2 Arbitrage diagnostics

On the reconstructed predicted surface, compute daily:

* mean negative calendar-violation mass
* mean negative butterfly-violation mass
* fraction of violating penalty-grid points

These diagnostics are reported for every model, every date, even if a model’s primary objective is not arbitrage reduction.

### 21.3 Diebold-Mariano tests

Use daily aggregated loss series and run pairwise **two-sided** DM tests for:

* squared IV error
* vega-weighted squared IV error

Implementation details:

* Newey-West HAC variance with bandwidth 5
* Harvey-Leybourne-Newbold finite-sample adjustment
* report raw and Holm-adjusted p-values

### 21.4 Model Confidence Set

Run the Hansen-Lunde-Nason MCS procedure on daily loss series for:

* `rmse_iv^2`
* `vega_rmse_iv^2`

Required outputs:

* `Tmax` and `TR` variants
* `alpha = 0.10`
* stationary bootstrap with 5000 resamples
* fixed bootstrap block length 10

## 22. Downstream utility diagnostics

The downstream diagnostics are part of the scientific method, not optional extras.

### 22.1 Mark-to-mid pricing utility

For every next-day contract with valid forward inputs and valid quotes:

1. query forecasted IV at the realized next-day contract point;
2. convert forecasted IV to a Black-Scholes price using:

   * realized next-day forward estimate,
   * realized next-day discount factor,
   * realized next-day `tau`,
   * contract strike and type;
3. compare predicted price with realized next-day 15:45 mid.

Report:

* `price_rmse`
* `price_mae`
* `inside_spread_rate`

### 22.2 Delta-hedged marking-error utility

For every eligible contract:

* hedge ratio is the **current-day vendor delta_15453**
* realized 1-day hedged PnL:

  * `(mid_{t+1} - mid_t) - delta_t * (S_{t+1} - S_t)`
* predicted 1-day hedged PnL:

  * `(price_hat_{t+1} - mid_t) - delta_t * (S_{t+1} - S_t)`

Here `S_t` is `active_underlying_price_15453`.

Report:

* hedged-PnL RMSE
* hedged-PnL MAE
* bucketed results for:

  * ATM call near 30d
  * ATM put near 30d
  * ATM call near 91d

This is the repository’s primary hedging-style diagnostic, motivated by the importance of hedging utility in Hull and White (2017) and related hedging papers in the corpus.

### 22.3 ATM straddle directional utility

For maturity anchors 30d and 91d:

1. identify nearest-ATM matched call-put pair on date `t`;
2. compute model signal from predicted ATM IV change `sigma_hat_{t+1}(0, tau*) - sigma_t(0, tau*)`;
3. go long the straddle if positive, short if negative;
4. mark the 1-day PnL from current 15:45 mid to next-day 15:45 mid;
5. subtract half-spread entry and exit costs on both legs.

Report:

* mean gross return
* mean net return
* hit rate
* Sharpe ratio on daily net returns

This is a stylized economic-value diagnostic, not a claim of deployable trading alpha.

## 23. Reproducibility controls

Every run must record:

* resolved config
* git commit
* Python version
* package versions
* CUDA availability and versions
* device used by each model family
* global random seed
* upstream artifact hashes

Seed policy:

* default global seed: `20260329`
* use the same seed for Python, NumPy, Torch, and XGBoost
* disable nondeterministic CuDNN behavior where applicable

Artifact immutability:

* artifact directories are immutable by default;
* rerunning with the same `run_id` without explicit overwrite is an error.

## 24. Performance, memory, and CUDA strategy

### 24.1 I/O and storage

* stream ZIP members directly;
* never materialize the full raw history as CSV on disk;
* write parquet partitioned by `underlying_key` and `year`;
* use lazy scans for downstream assembly.

### 24.2 Large-dataset rules

* no all-history pandas frames;
* no in-memory concatenation of all contracts across all years;
* batch reconstructor query generation by day;
* batch contract-level evaluation by day and by model;
* cache sampled-surface wide matrices once built.

### 24.3 CUDA policy

GPU-required stages:

* reconstructor training
* LSTM training/inference
* XGBoost training/inference

If CUDA is unavailable or a package is installed without GPU support for a GPU-required stage, the run must fail clearly.

CPU-only stages:

* inventory
* schema validation
* cleaning
* forward estimation
* DFW fitting
* PCA / VAR benchmark
* statistical testing

### 24.4 No silent downgrade rule

Never silently switch:

* GPU to CPU
* XGBoost to another booster
* PyTorch model to a different architecture
* parity-based forward estimation to a spot-moneyness fallback

## 25. Artifact contracts

The following artifacts are mandatory.

### 25.1 Inventory artifacts

* `raw_inventory.parquet`
* `raw_inventory.json`
* `vendor_schema_reconciliation.json`

### 25.2 Clean-data artifacts

* `clean_contracts.parquet`
* `forward_terms.parquet`
* `surface_nodes.parquet`

### 25.3 Surface artifacts

* `grid_definition.parquet`
* `sampled_surface_wide.parquet`

### 25.4 Modeling artifacts

* `features_targets.parquet`
* `split_manifest.json`
* `scalers.json`
* `reconstructor_model.pt`
* `reconstructor_manifest.json`
* model-family artifacts under `models/<model_name>/`

### 25.5 Evaluation artifacts

* `loss_panel.parquet`
* `arbitrage_panel.parquet`
* `pricing_utility.parquet`
* `hedged_pnl_utility.parquet`
* `straddle_signal_utility.parquet`
* `dm_tests.json`
* `mcs_results.json`
* `summary.md`

## 26. Test strategy

### 26.1 Unit tests

Required unit tests:

* filename/date discovery
* schema reconciliation
* contract cleaning
* parity forward estimation
* call-put collapse
* DFW fit and sampling
* feature window construction
* chronology split assertions
* PCA/VAR reconstruction
* DM statistic
* MCS bootstrap plumbing
* arbitrage-penalty tensor calculations

### 26.2 Integration tests

Required integration tests:

1. synthetic vendor-format dataset smoke run
2. full pipeline on a tiny date range with `spx_smoke.yaml`
3. deterministic rerun check for CPU stages
4. artifact-schema validation test

### 26.3 Acceptance criteria

The repository is done only when:

* all mandatory artifacts are produced by the canonical `run` command;
* all tests pass;
* chronology tests prove no future data enter any model input, scaler, PCA fit, or reconstructor fit;
* a smoke experiment completes end-to-end on the synthetic fixture;
* the README and AGENTS file match the implemented CLI and artifact structure.

## 27. Failure modes and explicit non-fallback behavior

The pipeline must stop with explicit errors for:

* missing raw root
* no matching `UnderlyingOptionsEODCalcs_*.zip` files
* unreadable ZIP
* multiple/no CSV inside ZIP
* missing required columns
* invalid selected underlying scope
* too few valid dates after preprocessing
* systematic forward-estimation failure
* singular DFW fit
* CUDA unavailable for GPU-required stages
* artifact collision without overwrite permission

The pipeline must **not**:

* silently skip failed stages and continue;
* auto-switch to alternative coordinates;
* auto-switch to CPU for GPU-required models;
* auto-change the grid;
* auto-add external data such as VIX, rates, or dividends.

## 28. Literature-to-design traceability

### 28.1 Surface representation

Chosen design:

* sampled fixed grid (SAM) on `m x tau`

Support:

* Cont and Da Fonseca (2002)
* Zhang, Li, and Zhang (2023)

Why this choice:

* strongest transparent representation in the core forecasting paper;
* simpler and more reproducible than latent-feature pipelines.

### 28.2 Reconstruction / completion

Chosen design:

* shared MLP reconstructor with static-arbitrage penalties

Support:

* Zhang, Li, and Zhang (2023)
* Ackerer, Tagasovska, and Vatter (2020)
* Chataigner, Crépey, and Dixon (2020)
* Wiedemann, Jacquier, and Gonon (2025)

### 28.3 Classical benchmark

Chosen design:

* PCA + VAR(1)

Support:

* Cont and Da Fonseca (2002)
* Fengler, Härdle, and Villa (2003)
* Chalamandaris and Tsekrekos (2010)

### 28.4 Nonlinear forecasting model set

Chosen design:

* XGBoost direct
* LSTM direct

Support:

* Audrino and Colangelo (2010)
* Medvedev and Wang (2022)
* Zhang, Li, and Zhang (2023)
* 2024 B-spline/tree IVS paper
* Arratia et al. (2025 leakage paper)

### 28.5 Split / leakage control

Chosen design:

* ordered train / validation / test blocks
* walk-forward tuning and test refits
* no random splits

Support:

* Chalamandaris and Tsekrekos (2010)
* Kearney, Shang, and Sheenan (2019)
* Shang and Kearney (2021)
* Cerqueira et al. (2020)
* Arratia et al. (2025)

### 28.6 Formal comparison tests

Chosen design:

* DM + MCS

Support:

* Diebold and Mariano (2002)
* Hansen, Lunde, and Nason (2011)

### 28.7 Downstream utility diagnostics

Chosen design:

* mark-to-mid pricing utility
* delta-hedged marking-error utility
* ATM straddle directional utility

Support:

* Bernales and Guidolin (2014)
* Chalamandaris and Tsekrekos (2010)
* Hull and White (2017)
* Vrontos, Galakis, and Vrontos (2021)

## 29. Ordered implementation plan for Codex

Codex must implement in this order:

1. repository scaffold, config system, logging, manifests
2. raw inventory + schema reconciliation
3. cleaned contract ingest for supported underlying
4. forward estimation and node collapse
5. DFW daily sampled-surface builder
6. feature/target dataset builder
7. chronology split engine
8. reconstructor with exact PDF-transcribed penalties
9. `rw_last` and `pca_var1`
10. `xgb_direct`
11. `lstm_direct`
12. contract-level evaluation + arbitrage diagnostics
13. DM + MCS
14. utility diagnostics
15. smoke tests, docs, and final polish

No stage may be replaced by an unreviewed alternative without revising this specification.
