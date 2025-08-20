from __future__ import annotations

from typing import Any


class Router:
    """
    Score-aware hard switch with:
      - dwell: minimum ticks to stay on current model
      - switch_threshold: minimum regime_score to permit switching
      - switch_penalty: extra margin required only when switching away from current
      - freeze_on_recent_cp: freeze decisions for `freeze_ticks` after cp_prob spike
    """
    def __init__(
        self,
        low_model: str = "arima",
        high_model: str = "xgb",
        dwell_min: int = 10,
        switch_threshold: float = 0.0,
        switch_penalty: float = 0.0,
        freeze_on_recent_cp: bool = False,
        freeze_ticks: int = 5,
        cp_spike_threshold: float = 0.6,
    ) -> None:
        self.low = low_model
        self.high = high_model
        self.dwell_min = int(dwell_min)
        self.switch_threshold = float(switch_threshold)
        self.switch_penalty = float(switch_penalty)
        self.freeze_on_recent_cp = bool(freeze_on_recent_cp)
        self.freeze_ticks = int(freeze_ticks)
        self.cp_spike_threshold = float(cp_spike_threshold)

        self._last: str | None = None
        self._dwell = 0
        self._freeze = 0

    def _want(self, regime_label: str) -> str:
        return self.high if regime_label == "high_vol" else self.low

    def choose(self, regime_label: str, regime_score: float, meta: dict[str, object] | None = None) -> str:
        m = meta or {}
        # Extract cp_prob robustly; fallback to regime_score if detector didn't provide it.
        raw: Any = m.get("cp_prob", regime_score)
        cp_prob = float(raw) if isinstance(raw, int | float) else float(regime_score)

        # freeze after a CP spike
        if self.freeze_on_recent_cp and cp_prob >= self.cp_spike_threshold:
            self._freeze = max(self._freeze, self.freeze_ticks)

        want = self._want(regime_label)

        # initial pick
        if self._last is None:
            self._last, self._dwell = want, 1
            if self._freeze > 0:
                self._freeze -= 1
            return want

        # freeze: hold current model
        if self._freeze > 0:
            self._freeze -= 1
            self._dwell += 1
            return self._last

        # staying with same model
        if want == self._last:
            self._dwell += 1
            return self._last

        # different target: enforce dwell
        if self._dwell < self.dwell_min:
            self._dwell += 1
            return self._last

        # require threshold + penalty to switch
        required = self.switch_threshold + self.switch_penalty
        if regime_score < required:
            self._dwell += 1
            return self._last

        # switch
        self._last = want
        self._dwell = 1
        return self._last
