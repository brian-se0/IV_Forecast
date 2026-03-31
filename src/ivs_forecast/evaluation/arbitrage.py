from __future__ import annotations

import numpy as np

from ivs_forecast.data.ssvi import static_arb_certification


def arbitrage_diagnostics(predicted_state: np.ndarray) -> dict[str, float | int | bool]:
    return static_arb_certification(predicted_state)
