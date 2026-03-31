# Literature Positioning

The implemented claim is intentionally narrow:

`A root-explicit, one-step, supervised, arbitrage-aware next-day IVS forecaster on Cboe Option EOD Summary data.`

This repo does not position itself as:

- a two-step sampled-grid plus reconstruction system
- a generic nonlinear forecasting novelty claim
- a generative IV surface model
- an operator-learning nowcasting system

Relative to nearby lines of work:

- Zhang et al.: this repo does not use a sampled-grid intermediate state plus a separate reconstruction network. The direct forecast target is the arbitrage-aware surface state itself.
- Neural-operator smoothing work: this repo is not framed as irregular-input nowcasting or smoothing of a same-day surface. It is a next-day forecasting pipeline with a fixed supervised target.
- NTK-style nonlinear forecasting: this repo does not claim novelty from nonlinearity alone. The emphasis is on the direct state-space and root-explicit data contract.
- VolGAN-style generation: this repo is not a generative model for sampling surfaces. It is a supervised next-day forecaster evaluated on realized target-day nodes and contracts.
