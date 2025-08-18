class Router:
    """Hard switch with a small dwell to reduce flip-flop."""
    def __init__(self, low_model: str = "arima", high_model: str = "xgb", dwell_min: int = 10) -> None:
        self.low = low_model
        self.high = high_model
        self.dwell_min = dwell_min
        self._last: str | None = None
        self._dwell = 0

    def choose(self, regime_label: str, regime_score: float) -> str:
        want = self.high if regime_label == "high_vol" else self.low
        if self._last is None:
            self._last, self._dwell = want, 1
            return want
        if want != self._last and self._dwell < self.dwell_min:
            self._dwell += 1
            return self._last
        if want != self._last:
            self._last, self._dwell = want, 1
        else:
            self._dwell += 1
        return self._last
