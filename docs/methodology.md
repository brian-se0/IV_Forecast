# Methodology

This repository implements one ordered workflow:

- verify and reconcile the vendor corpus
- write a raw-corpus contract for the discovered daily ZIP layout
- enforce exact requested-versus-observed window coverage for the canonical benchmark
- filter to `^SPX` and `option_root = "SPX"`
- clean contracts and compute fractional time to settlement with an explicit `AM_SOQ_PROXY` settlement policy
- estimate parity-implied forwards
- collapse root-explicit surface nodes
- calibrate one daily SSVI state
- build a trading-date continuity index and drop origin rows that would otherwise skip over a missing target day
- forecast the next day’s latent SSVI state
- decode forecasts directly onto realized next-day nodes and contracts

The implementation is deliberately one-step and direct. The predicted surface state is the model output space, and downstream evaluation queries that surface directly rather than routing through an alternate representation.
