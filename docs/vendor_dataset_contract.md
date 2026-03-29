# Vendor Dataset Contract

Supported raw files:

- `UnderlyingOptionsEODCalcs_YYYY-MM-DD.zip`

Required ZIP rules:

- exactly one CSV member;
- filename date must match `quote_date`;
- unreadable ZIPs, missing CSVs, or multiple CSV members are fatal.

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
