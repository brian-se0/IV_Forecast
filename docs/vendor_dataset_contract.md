# Vendor Dataset Contract

Supported raw files:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

Required ZIP rules:

- exactly one CSV member;
- filename date must match `quote_date`;
- unreadable ZIPs, missing CSVs, or multiple CSV members are fatal.
- `verify-data` audits every supported ZIP in the configured date window rather than sampling only one file.

Canonical required columns for this repository:

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
- underlying bid/ask fields

Observed-vs-documented note:

- local raw files and the vendor layout PDF use `*_1545`;
- any `*_15453` references are treated as stale documentation aliases and not as required columns.

Root-handling note:

- `root` is treated as the option-class key throughout forward estimation, node construction, and straddle pairing;
- the v1 sampled-surface path supports only one homogeneous root per modeling date and fails fast on mixed-root dates.

Audit caveats recorded by `verify-data`:

- quote-only files are unsupported and do not satisfy the v1 contract;
- missing `*_1545` calcs fields are fatal;
- zero or missing underlying bid/ask rates are summarized for the selected underlying;
- early-close dates are recorded from the in-repo curated half-day table as audit-only metadata.
