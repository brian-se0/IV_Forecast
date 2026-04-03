# Vendor Dataset Contract

Supported raw files:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

Discovery contract:

- canonical daily ZIPs may be discovered recursively under the configured raw root
- flat and nested daily directory layouts are supported
- grouped monthly/yearly archives must be detected explicitly and reported in `raw_corpus_contract.json`
- the canonical benchmark requires the requested start/end dates to match the observed daily-ZIP window exactly

Fatal ZIP conditions:

- unreadable ZIP
- zero CSV members
- multiple CSV members
- filename date does not match `quote_date`

Required columns:

- `underlying_symbol`
- `quote_date`
- `root`
- `expiration`
- `strike`
- `option_type`
- `bid_1545`
- `ask_1545`
- `active_underlying_price_1545`
- `implied_volatility_1545`
- `delta_1545`
- `vega_1545`
- `trade_volume`
- `open_interest`

Optional but ingested when present:

- `gamma_1545`
- `theta_1545`
- `rho_1545`
- `bid_eod`
- `ask_eod`
- `vwap`
- `open`
- `high`
- `low`
- `close`

Ignored for modeling:

- `implied_underlying_price_1545`
- `delivery_code`
- quote-size columns
- underlying bid/ask columns beyond audit summaries

Contract notes:

- the implementation treats `root` as the option-class key
- the validated v1 path filters to `option_root = "SPX"` before modeling artifacts are written
- the `1545` suffix remains authoritative even on early-close days
- early-close dates come from the checked-in `cboe_us_options_early_closes.csv` manifest
- zero IV rows are invalid surface observations
- quote-only files are unsupported

`verify-data` writes a machine-readable reconciliation report and includes per-date root coverage for the configured `underlying_symbol` and `option_root`.
For the canonical benchmark, any requested/observed window mismatch is a fatal verification error after the coverage contract is written.
