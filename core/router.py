from __future__ import annotations


class Router:
    """Hard switch with dwell and optional score threshold to block low-confidence flips."""
    def __init__(self, low_model: str = "arima", high_model: str = "xgb",
                 dwell_min: int = 10, switch_threshold: float = 0.0,
                 freeze_on_recent_cp: bool = False) -> None:
        self.low = low_model
        self.high = high_model
        self.dwell_min = int(dwell_min)
        self.switch_threshold = float(switch_threshold)
        self.freeze_on_recent_cp = bool(freeze_on_recent_cp)
        self._last: str | None = None
        self._dwell = 0

    def choose(self, regime_label: str, regime_score: float, meta: dict | None = None) -> str:
        want = self.high if regime_label == "high_vol" else self.low
        if self._last is None:
            self._last, self._dwell = want, 1
            return want

        # Optional freeze right after a CP spike
        if self.freeze_on_recent_cp and meta and meta.get("recent_cp", False):
            self._dwell += 1
            return self._last

        # Block switching on low confidence
        if want != self._last and regime_score < self.switch_threshold:
            self._dwell += 1
            return self._last

        # Dwell logic
        if want != self._last and self._dwell < self.dwell_min:
            self._dwell += 1
            return self._last

        if want != self._last:
            self._last, self._dwell = want, 1
        else:
            self._dwell += 1
        return self._last
