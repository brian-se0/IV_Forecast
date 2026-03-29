# Methodology

This repository implements the ordered workflow defined in `implementation_spec.md`.

Version 1 uses:

- one supported underlying per experiment;
- index-underlying workflow only;
- default validated symbol `^SPX`;
- next-valid-day forecasting at the 15:45 ET surface;
- a fixed 154-point sampled surface in log-forward-moneyness and maturity;
- a shared arbitrage-aware reconstructor implemented from the source-paper PDF equations.

Important implementation note:

- the authoritative observed/vendor dataset schema in this repository is the `*_1545` field family;
- reconciliation explicitly records this against any stale documentation that refers to alternate suffixes.
