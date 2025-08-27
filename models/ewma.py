from __future__ import annotations

from typing import Any


class EWMAModel:
    """
    Tiny online EWMA forecaster.
    Predicts the next tick as the current EMA; then updates EMA with the current x.
    No leakage: prediction uses EMA before seeing the next truth.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
        self._ema: float | None = None

    def predict_update(
        self, tick: dict[str, Any], feats: dict[str, float]
    ) -> tuple[float, dict[str, Any]]:
        x = float(tick["x"])
        y_hat = 0.0 if self._ema is None else float(self._ema)
        self._ema = x if self._ema is None else self.alpha * x + (1.0 - self.alpha) * self._ema
        return y_hat, {}

