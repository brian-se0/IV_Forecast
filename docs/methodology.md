# Methodology

This repository implements one ordered workflow:

- verify and reconcile the vendor corpus
- filter to `^SPX` and `option_root = "SPX"`
- clean contracts and compute fractional time to settlement
- estimate parity-implied forwards
- collapse root-explicit surface nodes
- calibrate one daily SSVI state
- forecast the next day’s latent SSVI state
- decode forecasts directly onto realized next-day nodes and contracts

The implementation is deliberately one-step and direct. The predicted surface state is the model output space, and downstream evaluation queries that surface directly rather than routing through an alternate representation.
