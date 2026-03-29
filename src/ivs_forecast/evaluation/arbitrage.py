from __future__ import annotations

import numpy as np

from ivs_forecast.models.reconstructor import FittedReconstructor


def arbitrage_diagnostics(
    reconstructor: FittedReconstructor,
    sampled_surface_iv: np.ndarray,
) -> dict[str, float]:
    return reconstructor.arbitrage_diagnostics(sampled_surface_iv)
