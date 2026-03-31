Supplemental literature used

* Chen et al., **“Neural Tangent Kernel in Implied Volatility Forecasting”** (*Journal of Business & Economic Statistics*, 2026). I used this to map the current nonlinear IVS-forecasting frontier, so the new plan does not mistake “nonlinear forecasting” by itself for novelty. ([IDEAS/RePEc][1])
* Vuletić and Cont, **“VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces”** (*Applied Mathematical Finance*, 2024). I used this to rule out a generic “arbitrage-free generative deep model” as a sufficient novelty claim. ([IDEAS/RePEc][2])
* Arratia et al., **“Examining Challenges in Implied Volatility Forecasting”** (*Computational Economics*, 2025). I used this to reinforce that grouped chronological splits and explicit leakage control remain mandatory. ([Springer][3])

## Implementation plan for GPT-5.4 in Codex IDE

I reviewed commit `8674d71ded83ff1eaa9492346228c4a12c534118`. The current repo is still organized around the old two-step forecasting heart: build a 154-point sampled surface with DFW, forecast that sampled surface with `rw_last`, `pca_var1`, `xgb_direct`, and `lstm_direct`, then use a shared no-arbitrage reconstructor during validation/test-time evaluation. `StudyConfig` still lacks an experiment-level `option_root`, while the vendor layout defines `root` as the option trading class symbol and the dataset itself as a 15:45-plus-EOD daily product with calc fields only in calcs-included files. The good news is that the repo already has several strong pieces worth preserving: strict corpus auditing, root-aware forward estimation, root-aware node collapse, partitioned raw/intermediate artifacts, chronology-safe split logic, and DM/MCS utilities. ([GitHub][4]) 

The right hard cutover is **not** “put a different deep model inside the same Zhang-style sampled-surface-plus-reconstructor pipeline.” Zhang et al. already occupy that lane: Step 1 predicts extracted surface features, and Step 2 reconstructs the full surface with a DNN under static-arbitrage penalties; they also explicitly name exogenous variables like index return and VIX as an obvious next extension. At the same time, newer literature already covers nonlinear functional IVS forecasting, arbitrage-free generative IVS modeling, and irregular-input neural-operator smoothing. So the cleanest finishable path is a **direct, one-step, supervised, root-explicit, arbitrage-aware next-day surface forecaster**, with the arbitrage-safe surface itself as the model output space rather than a sampled IV grid that later needs reconstruction.     ([IDEAS/RePEc][1])

My recommendation is a **direct SSVI-state forecasting cutover**:

* keep the existing vendor audit, ingestion, cleaning, parity-forward estimation, node construction, split protocol, statistical testing, and downstream utility diagnostics;
* delete the sampled-surface / reconstructor architecture entirely;
* add an explicit `option_root` contract and root-aware time-to-settlement logic;
* calibrate one arbitrage-free SSVI surface state per date;
* train a single deep temporal model to forecast the next-day SSVI state directly;
* evaluate by decoding the predicted SSVI state straight onto the realized next-day node and contract coordinates.

This is materially different from Zhang’s two-step framework, much safer than RL, and realistic to finish. The remaining empirical step after implementation would be the full Windows/CUDA run and inspection of the evidence pack.   ([IDEAS/RePEc][1])

## 1. Hard-cutover target

The end-state codebase should implement exactly one research protocol:

**`^SPX`, root-explicit, one-step, arbitrage-free, next-day implied-volatility-surface forecasting via direct SSVI-state prediction.**

Not “one of several methods.” Not “legacy plus new option.” One protocol.

Use the Cboe Option EOD Summary 15:45 calc snapshot as the modeling timestamp, preserve the existing fail-fast vendor-audit discipline, and make the modeled option class explicit through `option_root`. The vendor layout PDF is unambiguous that `root` is a first-class contract field, and the dataset is centered on the 15:45 calc snapshot. 

For the canonical v1 path, support only:

* `underlying_symbol = "^SPX"`
* `option_root = "SPX"`
* forecast horizon = 1 trading day
* max maturity = 730 calendar days
* calcs-included Cboe files only

That narrower scope is a feature, not a bug. It solves the contract-class problem cleanly and keeps the repo focused on one coherent publication-grade protocol. Cboe’s official SPX materials also distinguish AM-settled traditional `SPX` from PM-settled `SPXW`, and state that both can coexist on standard expiries. Narrowing v1 to `SPX` avoids a large amount of avoidable ambiguity. ([CBOE][5])

## 2. Hard-cutover rules for Codex

Codex should follow these rules exactly.

First, create a git tag for the current branch head before deleting anything, for example `archive/zhang-two-step-8674d71`. Keep that only as git history, not as live runtime code.

Second, remove every legacy forecasting path as soon as the new one compiles. That means no backwards compatibility layer, no feature flag, and no dual representation support.

Third, do not add new external market datasets such as VIX feeds. Zhang already suggests index return and VIX as future inputs, so adding them would not solve the novelty problem and would increase operational scope. Keep v1 self-contained in the Cboe dataset plus the repo’s own derived quantities. 

Fourth, do not introduce RL, trading policy optimization, or simulator logic. This project remains a forecasting paper with downstream utility diagnostics, not a sequential decision paper.

Fifth, do not keep XGBoost just to preserve an old baseline. The current `pyproject.toml` still depends on `xgboost`, but once the sampled-surface baselines are deleted that dependency should go too. The current Python 3.14 toolchain can stay: official PyTorch’s Windows support matrix covers Python 3.10–3.14, and PyTorch 2.10 adds Python 3.14 `torch.compile()` support. ([GitHub][6])

## 3. New methodology to implement

### 3.1 Representation: direct arbitrage-free SSVI state

Replace the current 154-point sampled-surface representation with a daily arbitrage-free **SSVI state**.

Use total variance, not IV, as the canonical surface representation:

`w(k, τ) = 0.5 * θ(τ) * [1 + ρ * φ(θ(τ)) * k + sqrt((φ(θ(τ)) * k + ρ)^2 + 1 - ρ^2)]`

with

`φ(θ) = η * θ^{-λ}`

and constrained parameters:

* `θ(τ) > 0`, monotone nondecreasing in `τ`
* `ρ ∈ (-1, 1)`
* `η > 0`
* `λ ∈ [0, 0.5]`

This is grounded in the established arbitrage-free SVI/SSVI literature rather than the Zhang-style sampled-grid-plus-reconstructor setup.  ([Taylor & Francis Online][7])

Use the repo’s current maturity support as the SSVI ATM knot grid:

`[10, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730]` days.

Those 11 maturities are already the canonical maturity anchors in the repo’s current DFW grid; reusing them keeps domain coverage stable while deleting the obsolete 154-grid representation. ([GitHub][8])

The daily state vector should therefore be:

* 11 monotone ATM total-variance knot values `theta_d010 ... theta_d730`
* 1 global `rho`
* 1 global `eta`
* 1 global `lambda`

Total dimension: 14.

Also store an unconstrained latent vector `state_z_000 ... state_z_013` as the canonical model target:

* `z_theta = log(theta_knot)`
* `z_rho = atanh(rho)`
* `z_eta = log(eta)`
* `z_lambda = logit(2 * lambda)`

All forecasting models will predict `state_z`; all decoding/evaluation will use the constrained transformed parameters.

### 3.2 Loss space: train on target-day nodes, not parameter-vector MSE

Do **not** train the deep model to minimize MSE between forecasted and realized parameter vectors. That creates avoidable identifiability and scaling problems.

Instead:

* forecast the next-day latent SSVI state,
* transform it into valid constrained parameters,
* decode the predicted surface onto the realized target-day node coordinates `(m, τ)`,
* compute the training loss directly against the realized target-day node IVs.

The primary training loss should be **vega-weighted Huber loss** on target-day nodes. This keeps the model aligned with the actual forecasting objective and avoids overfitting to an arbitrary param-coordinate geometry.

### 3.3 Deep model: one canonical architecture

Implement one deep model only:

**`ssvi_tcn_direct`**

Use a temporal convolutional network over a fixed 22-trading-day history of the daily SSVI state plus scalar regime/liquidity features.

Architecture:

* input sequence length: 22
* per-day feature vector: 14 latent state dims + scalar features
* projection layer to width 64
* 4 residual causal 1D-conv blocks
* kernel size 3
* dilations `[1, 2, 4, 8]`
* GELU activations
* dropout `0.1`
* LayerNorm after projection and inside blocks
* MLP head from final hidden state to 14 latent outputs

This is deliberately simple, fast, and GPU-friendly. The novelty claim should come from the **direct arbitrage-free forecasting formulation**, not from choosing a flashy network. Recent literature already occupies “nonlinear IVS forecasting” and “irregular-input smoothing,” so the architecture should optimize for finishability and stable runtime. ([IDEAS/RePEc][1]) 

### 3.4 Baselines after cutover

Keep exactly two simple baselines in the new representation:

* `state_last`: tomorrow’s latent SSVI state = today’s latent SSVI state
* `state_var1`: OLS VAR(1) with intercept on the latent SSVI state vector

Delete `rw_last`, `pca_var1`, `xgb_direct`, `lstm_direct`, and the reconstructor path entirely. The old baselines are representation-specific to the deleted sampled-surface workflow. The new baselines should live in the same output manifold as the new method.

## 4. Exact implementation phases

### Phase 0: archive old baseline, then delete legacy method code

**Why:** this is the only way to satisfy the hard-cutover requirement.

**Files to delete after tagging the old state**

* `src/ivs_forecast/data/dfw.py`
* `src/ivs_forecast/data/sampled_surface.py`
* `src/ivs_forecast/models/reconstructor.py`
* `src/ivs_forecast/models/rw_last.py`
* `src/ivs_forecast/models/pca_var1.py`
* `src/ivs_forecast/models/xgb_direct.py`
* `src/ivs_forecast/models/lstm_direct.py`
* `configs/models/reconstructor.yaml`
* `configs/models/reconstructor_smoke.yaml`
* `configs/models/lstm.yaml`
* `configs/models/lstm_smoke.yaml`
* `configs/models/xgboost.yaml`
* `configs/models/xgboost_smoke.yaml`

**Also remove**

* all imports and branching logic referencing those modules in `train_models.py`, `forecast.py`, `run_experiment.py`, tests, docs, and README.

**Acceptance criteria**

* ripgrep over `src/`, `tests/`, and `docs/` returns no matches for:

  * `sampled_surface`
  * `reconstructor`
  * `rw_last`
  * `pca_var1`
  * `xgb_direct`
  * `lstm_direct`

### Phase 1: make the experiment contract root-explicit and settlement-time-correct

**Why:** the current repo has root-aware node construction but still no experiment-level `option_root`, and `clean.py` still uses a simple `dte_days / 365` time-to-expiry. For `SPX` versus `SPXW`, Cboe’s materials show that settlement conventions differ materially. ([GitHub][9])

**Files to modify**

* `src/ivs_forecast/config.py`
* `src/ivs_forecast/data/clean.py`
* `src/ivs_forecast/data/discovery.py`
* `src/ivs_forecast/pipeline/build_data.py`
* `configs/experiments/spx_1d.yaml`
* `configs/experiments/spx_smoke.yaml`
* `README.md`
* `implementation_spec.md`
* `docs/vendor_dataset_contract.md`
* `docs/artifact_contracts.md`

**Actions**

1. Add required `option_root: Literal["SPX"]` to `StudyConfig` for v1.
2. Filter to the configured root immediately after ingestion and before any modeling artifact is built.
3. Add root counts by date to `verify-data` and write a root coverage section in the audit report.
4. Replace integer-day `tau` with **fractional calendar-time-to-settlement**:

   * snapshot timestamp = 15:45 ET on normal days, 12:45 ET on early-close days;
   * settlement timestamp for `SPX` = opening-settlement session on expiration date.
5. Persist `option_root` in every downstream artifact, including manifests and summary reports.

**New helper module**

* `src/ivs_forecast/data/time_to_settlement.py`

Implement:

* `snapshot_timestamp_eastern(quote_date, is_early_close)`
* `settlement_timestamp_eastern(expiration, option_root)`
* `year_fraction_act365(start_ts, end_ts)`

Use `zoneinfo` and built-in datetime utilities; do not add a heavy timezone dependency.

**Acceptance criteria**

* `StudyConfig` cannot be instantiated without `option_root`.
* `build-data` fails if any post-filter modeling date contains zero rows for the configured root.
* all modeling artifacts contain `option_root`.
* unit tests verify that `SPX` uses AM settlement timing and that early-close days shift the snapshot to 12:45 ET.  ([CBOE][5])

### Phase 2: add daily SSVI calibration and surface-state artifacts

**Why:** this is the core representation cutover. It removes the sampled-surface/reconstructor dependency and gives the forecasting model an arbitrage-aware output manifold.

**Files to create**

* `src/ivs_forecast/data/ssvi.py`

This module should contain:

* `maturity_knots_days()`
* `maturity_knots_tau()`
* `raw_to_constrained_params(raw_tensor)`
* `constrained_to_raw_params(params_tensor)`
* `theta_curve(tau_query, theta_knots)`
* `ssvi_total_variance(m, tau, params)`
* `ssvi_implied_vol(m, tau, params)`
* `static_arb_certification(params, m_grid, tau_grid, tol)`
* `initial_params_from_nodes(nodes)`
* `calibrate_daily_ssvi(nodes, init_raw, config)`

**Files to modify**

* `src/ivs_forecast/pipeline/build_data.py`
* `src/ivs_forecast/data/collapse_nodes.py`
* `docs/artifact_contracts.md`
* `README.md`
* `implementation_spec.md`

**Actions**

1. Keep `clean_contracts`, `forward_terms`, and `surface_nodes` exactly as the pre-modeling data backbone.
2. After `surface_nodes` are built, add a new stage:

   * group by `quote_date`
   * calibrate one SSVI state per date
   * certify static arbitrage on a dense grid
   * write:

     * `ssvi_state.parquet`
     * `ssvi_fit_diagnostics.parquet`
     * `ssvi_certification.parquet`
3. Delete `sampled_surface_wide.parquet` and `grid_definition.parquet` from the artifact contract.

**Calibration details**

* fit in total-variance space or IV space consistently; I recommend vega-weighted Huber loss on IV because the downstream metrics are IV-based;
* initialize `theta` from near-ATM nodes by maturity, made monotone by cumulative maximum before optimization;
* use one deterministic optimizer schedule for every date:

  * 50 Adam steps
  * then 75 LBFGS steps
* no fallback optimizer branches;
* warm-start each day from the prior date’s solution.

**Certification details**

* certify on a dense grid spanning the current repo support:

  * moneyness ratios `0.60` to `2.00`
  * maturities `10` to `730` days, with midpoints inserted between anchor maturities
* write max negative calendar residual, max negative butterfly residual, and violation counts.

**Acceptance criteria**

* synthetic recovery test: fit a generated SSVI surface and recover it with low error;
* certification test: generated arbitrage-free surfaces pass certification to numerical tolerance;
* build-data on the synthetic smoke corpus writes the three new SSVI artifacts and no sampled-surface artifacts.

### Phase 3: replace the feature panel with a lean state-sequence panel

**Why:** the current `features_targets.parquet` duplicates a large fixed-grid representation. That should go away.

**Files to modify**

* `src/ivs_forecast/features/dataset.py`
* `src/ivs_forecast/features/scalars.py`
* `src/ivs_forecast/models/base.py`

**Actions**

1. Rewrite `features/dataset.py` so the canonical model input is derived from `ssvi_state.parquet`, not sampled surfaces.
2. Keep scalar features that are already chronology-safe and internal to the Cboe data:

   * underlying log return 1d / 5d / 22d
   * log total trade volume
   * log total open interest
   * median relative spread
   * surface fit RMSE from the current day’s calibration
   * node count and valid-expiry count
3. Build a narrow `features_targets.parquet` with:

   * `quote_date`
   * `target_date`
   * `option_root`
   * `history_start_index`
   * `history_end_index`
   * scalar features for the origin date
   * `surface_state_row_index`
   * `target_state_row_index`
4. Do **not** materialize lagged copies of the state vector in the parquet file. Sequence assembly should happen in the PyTorch dataset class from `ssvi_state.parquet`.

**Chronology rules**

* minimum history = 22 trading days
* target = next available modeling date
* no centered windows
* no normalization fitted on full sample

**Acceptance criteria**

* a unit test proves that `target_date` is always the next available date;
* a leakage test proves that a future-only synthetic shock cannot enter the origin-date feature row;
* `features_targets.parquet` stays narrow and root-explicit.

### Phase 4: implement the new model set

**Files to create**

* `src/ivs_forecast/models/state_last.py`
* `src/ivs_forecast/models/state_var1.py`
* `src/ivs_forecast/models/ssvi_tcn_direct.py`

**Files to modify**

* `src/ivs_forecast/models/base.py`
* `src/ivs_forecast/pipeline/train_models.py`

**Model definitions**

#### `state_last`

No fitting. Forecast next-day latent state as current latent state.

#### `state_var1`

OLS VAR(1) with intercept on the latent state vector only. Fit on training rows. Forecast latent state, then transform to constrained SSVI parameters.

#### `ssvi_tcn_direct`

PyTorch model only.

**Dataset behavior**

Each batch item should contain:

* origin-date 22-day history tensor of `[latent_state | scalar_features]`
* target-date padded node arrays:

  * `m`
  * `tau`
  * `node_iv`
  * `node_vega`
  * mask

Use `pad_sequence` in the collate function.

**Training loss**

Primary:

* vega-weighted Huber loss on target nodes

Secondary:

* none unless training is numerically unstable

Do not add arbitrary auxiliary losses unless needed.

**Training loop**

* optimizer: AdamW
* lr: `3e-4`
* weight decay: `1e-4`
* max epochs: `150`
* early stopping patience: `15`
* gradient clipping: `1.0`
* batch size: `64`
* deterministic seeds preserved in manifests

**Tuning grid**

Keep it intentionally small:

* `history_days`: `[10, 22]`
* `hidden_width`: `[32, 64]`
* `dropout`: `[0.0, 0.1]`

Select by validation vega-weighted RMSE only.

**Acceptance criteria**

* unit test that model outputs always map to valid constrained SSVI params;
* one synthetic training-step test showing the loss decreases on an easy synthetic panel;
* one artifact test confirming saved model checkpoints include the chosen hyperparameters and normalization statistics.

### Phase 5: rewrite the experiment pipeline around direct surface-state forecasting

**Files to modify**

* `src/ivs_forecast/pipeline/train_models.py`
* `src/ivs_forecast/pipeline/forecast.py`
* `src/ivs_forecast/pipeline/run_experiment.py`
* `src/ivs_forecast/cli.py`

**Actions**

1. Replace the old `MODEL_FAMILIES` list with:

   * `state_last`
   * `state_var1`
   * `ssvi_tcn_direct`
2. Remove all reconstructor training, loading, and artifact writing.
3. Validation and test forecasting should produce predicted latent states, then constrained SSVI parameters.
4. Decode predictions directly to:

   * target-day surface nodes for loss metrics
   * target-day clean contracts for pricing/hedging/straddle utilities
5. Keep the walk-forward refit logic from the current repo.
6. Keep DM and MCS exactly as post-fix repo utilities; only the loss panel source changes.

**New forecast artifacts**

Per model, write:

* `forecast_ssvi_state.parquet`
* `forecast_node_panel.parquet`
* `forecast_contract_panel.parquet`

Run-level outputs should still include:

* `loss_panel.parquet`
* `pricing_utility.parquet`
* `hedged_pnl_utility.parquet`
* `straddle_signal_utility.parquet`
* `dm_tests.json`
* `mcs_results.json`
* `summary.md`

Add:

* `forecast_ssvi_certification.parquet`

**Acceptance criteria**

* the run pipeline contains no reconstructor branches;
* all evaluation utilities are fed by direct SSVI queries;
* the summary report prints `underlying_symbol`, `option_root`, and the new representation family.

### Phase 6: update evaluation, reporting, and novelty positioning

**Files to modify**

* `src/ivs_forecast/evaluation/arbitrage.py`
* `README.md`
* `implementation_spec.md`
* `docs/artifact_contracts.md`
* `docs/vendor_dataset_contract.md`

**Files to create**

* `docs/literature_positioning.md`

**Actions**

1. Update `evaluation/arbitrage.py` so the diagnostics consume dense decoded SSVI surfaces, not reconstructor outputs.
2. Rewrite the README and implementation spec so they no longer describe the project as following Zhang’s sampled-surface plus Step-2 reconstructor design.
3. Add a short literature-positioning memo that compares the new design against:

   * Zhang et al. two-step arbitrage-aware forecasting
   * operator deep smoothing for irregular-input nowcasting
   * NTK nonlinear IVS forecasting
   * VolGAN arbitrage-free generation

The public claim should be narrow and supportable:

**“A root-explicit, one-step, supervised, arbitrage-aware next-day IVS forecaster on Cboe Option EOD Summary data.”**

Do **not** claim “first” anywhere unless the final literature sweep after the full run justifies it.

**Acceptance criteria**

* no README/spec text says the method “follows Zhang et al. (2023)” as the core design;
* the summary template includes a literature-positioning paragraph and a root/snapshot policy paragraph.

### Phase 7: test suite, smoke coverage, and repo cleanup

**Files to create**

* `tests/unit/test_ssvi_surface.py`
* `tests/integration/test_direct_ssvi_pipeline.py`

**Files to modify**

* `tests/unit/test_data_pipeline.py`
* `tests/unit/test_features_and_stats.py`
* `tests/unit/test_artifacts_and_mcs.py`
* `tests/integration/test_cli_pipeline.py`
* `pyproject.toml`

**Actions**

1. Remove `xgboost` from `pyproject.toml`.
2. Keep the current PyTorch CUDA wheel configuration.
3. Add the following tests:

**Vendor/root/time tests**

* root filter enforced
* `option_root` required
* fractional `tau` differs correctly between normal and early-close dates
* SPX settlement timing applied

**SSVI tests**

* synthetic parameter recovery
* certification on generated surfaces
* decode-to-IV monotonicity / positivity sanity checks

**Chronology tests**

* next-date target alignment
* no future leakage in sequence assembly
* normalization fit only on training rows

**Model tests**

* `state_var1` artifact shape test
* `ssvi_tcn_direct` forward-pass and masked-loss test
* synthetic easy-panel overfit test

**CLI smoke**

* `verify-data`
* `build-data`
* `run` on a tiny synthetic corpus

The full GPU run can stay skipped when CUDA is unavailable, but the rest of the test suite should remain CPU-runnable.

**Acceptance criteria**

* `uv run --active pytest -q` passes except the explicitly CUDA-gated full run
* `uv run --active ruff check src tests` passes
* `uv run --active python -c "import ivs_forecast"` passes
* `uv run --active ivs-forecast --help` passes

## 5. Exact file map

### Delete

* `src/ivs_forecast/data/dfw.py`
* `src/ivs_forecast/data/sampled_surface.py`
* `src/ivs_forecast/models/reconstructor.py`
* `src/ivs_forecast/models/rw_last.py`
* `src/ivs_forecast/models/pca_var1.py`
* `src/ivs_forecast/models/xgb_direct.py`
* `src/ivs_forecast/models/lstm_direct.py`
* old model configs for reconstructor / LSTM / XGBoost

### Create

* `src/ivs_forecast/data/time_to_settlement.py`
* `src/ivs_forecast/data/ssvi.py`
* `src/ivs_forecast/models/state_last.py`
* `src/ivs_forecast/models/state_var1.py`
* `src/ivs_forecast/models/ssvi_tcn_direct.py`
* `tests/unit/test_ssvi_surface.py`
* `tests/integration/test_direct_ssvi_pipeline.py`
* `docs/literature_positioning.md`

### Rewrite substantially

* `src/ivs_forecast/config.py`
* `src/ivs_forecast/data/clean.py`
* `src/ivs_forecast/data/discovery.py`
* `src/ivs_forecast/features/dataset.py`
* `src/ivs_forecast/features/scalars.py`
* `src/ivs_forecast/models/base.py`
* `src/ivs_forecast/pipeline/build_data.py`
* `src/ivs_forecast/pipeline/train_models.py`
* `src/ivs_forecast/pipeline/forecast.py`
* `src/ivs_forecast/pipeline/run_experiment.py`
* `src/ivs_forecast/evaluation/arbitrage.py`
* `README.md`
* `implementation_spec.md`
* `docs/artifact_contracts.md`
* `docs/vendor_dataset_contract.md`
* `pyproject.toml`

## 6. New artifact contract

After cutover, the canonical build/run artifacts should be:

Build stage:

* `vendor_schema_reconciliation.json`
* `data_audit_report.md`
* `clean_contracts/`
* `clean_contracts_index.parquet`
* `surface_nodes/`
* `surface_nodes_index.parquet`
* `forward_terms.parquet`
* `ssvi_state.parquet`
* `ssvi_fit_diagnostics.parquet`
* `ssvi_certification.parquet`
* `features_targets.parquet`
* stage manifests

Run stage:

* `split_manifest.json`
* per-model:

  * `forecast_ssvi_state.parquet`
  * `forecast_node_panel.parquet`
  * `forecast_contract_panel.parquet`
* run-level:

  * `loss_panel.parquet`
  * `forecast_ssvi_certification.parquet`
  * `pricing_utility.parquet`
  * `hedged_pnl_utility.parquet`
  * `straddle_signal_utility.parquet`
  * `dm_tests.json`
  * `mcs_results.json`
  * `summary.md`
  * stage manifests

Artifacts that should no longer exist:

* `grid_definition.parquet`
* `sampled_surface_wide.parquet`
* reconstructor checkpoints/manifests
* any sampled-surface forecast artifacts

## 7. Recommended commit sequence for Codex

Use six reviewable commits.

Commit 1: root-explicit contract + settlement-time fix + docs skeleton
Commit 2: SSVI calibration stage + new artifacts + synthetic SSVI tests
Commit 3: dataset rewrite + `state_last` + `state_var1`
Commit 4: `ssvi_tcn_direct` + training loader + validation tuning
Commit 5: run/evaluation/report cutover + remove reconstructor path
Commit 6: delete legacy modules/configs, remove `xgboost`, finish docs/tests/README cleanup

## 8. Final acceptance checklist

When Codex is done, these must all be true before you touch the Windows CUDA machine:

* the repo no longer contains sampled-surface or reconstructor runtime code;
* the canonical experiment config requires `option_root: "SPX"` and carries that value into all artifacts;
* time-to-settlement is fractional and root-aware, not `dte_days / 365`;
* `build-data` produces `ssvi_state.parquet` and no sampled-surface artifacts;
* the only model families are `state_last`, `state_var1`, and `ssvi_tcn_direct`;
* the deep model is trained against target-day node loss, not parameter-vector MSE;
* the evaluation path decodes direct SSVI forecasts onto realized next-day nodes/contracts;
* DM/MCS still run on the new loss panel;
* README, implementation spec, and artifact docs all describe the new one-step direct methodology;
* clean-clone import, lint, and test commands pass;
* the only remaining major step is the full real-corpus Windows/CUDA run and inspection of the evidence pack.

If Codex implements this plan faithfully, the codebase should no longer be structurally blocked by the Zhang-style novelty problem, and the only substantive thing left will be the final end-to-end execution and evidence review.

[1]: https://ideas.repec.org/a/taf/jnlbes/v44y2026i1p24-38.html "https://ideas.repec.org/a/taf/jnlbes/v44y2026i1p24-38.html"
[2]: https://ideas.repec.org/a/taf/apmtfi/v31y2024i4p203-238.html "https://ideas.repec.org/a/taf/apmtfi/v31y2024i4p203-238.html"
[3]: https://link.springer.com/article/10.1007/s10614-025-11172-z "https://link.springer.com/article/10.1007/s10614-025-11172-z"
[4]: https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/README.md?utm_source=chatgpt.com "raw.githubusercontent.com"
[5]: https://cdn.cboe.com/resources/spx/spx-fact-sheet.pdf?utm_source=chatgpt.com "SPX® Index Options"
[6]: https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/pyproject.toml "https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/pyproject.toml"
[7]: https://www.tandfonline.com/doi/abs/10.1080/14697688.2013.819986 "https://www.tandfonline.com/doi/abs/10.1080/14697688.2013.819986"
[8]: https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/src/ivs_forecast/data/dfw.py "https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/src/ivs_forecast/data/dfw.py"
[9]: https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/src/ivs_forecast/config.py "https://raw.githubusercontent.com/brian-se0/IV_Forecast/8674d71ded83ff1eaa9492346228c4a12c534118/src/ivs_forecast/config.py"
