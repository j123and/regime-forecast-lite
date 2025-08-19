from __future__ import annotations

from collections import deque


def _weighted_quantile(vals: list[float], wts: list[float], q: float) -> float:
    assert 0.0 <= q <= 1.0
    if not vals:
        return 0.0
    pairs = sorted((abs(v), float(w)) for v, w in zip(vals, wts, strict=False) if w > 0)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    cutoff = q * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= cutoff:
            return float(v)
    return float(pairs[-1][0])

class OnlineConformal:
    """
    Absolute-residual conformal with:
      - sliding window + exponential decay
      - optional per-regime buffers (e.g., 'low_vol', 'high_vol')
      - multi-α interval computation
    """
    def __init__(
        self,
        window: int = 500,
        decay: float = 1.0,
        by_regime: bool = False,
        cold_scale: float = 0.01,
    ) -> None:
        self.window = int(window)
        self.decay = float(decay)
        self.by_regime = bool(by_regime)
        self.cold_scale = float(cold_scale)

        self.res_global: deque[float] = deque(maxlen=self.window)
        self.wts_global: deque[float] = deque(maxlen=self.window)

        self._res_by_regime: dict[str, deque[float]] = {}
        self._wts_by_regime: dict[str, deque[float]] = {}

    def _decay(self, wts: deque[float]) -> None:
        if self.decay < 1.0 and wts:
            for i in range(len(wts)):
                wts[i] *= self.decay

    def _buffers_for(self, regime: str | None) -> tuple[deque[float], deque[float]]:
        if not self.by_regime or not regime:
            return self.res_global, self.wts_global
        if regime not in self._res_by_regime:
            self._res_by_regime[regime] = deque(maxlen=self.window)
            self._wts_by_regime[regime] = deque(maxlen=self.window)
        return self._res_by_regime[regime], self._wts_by_regime[regime]

    def update(self, y_hat: float, y_true: float, regime_label: str | None = None) -> None:
        self._decay(self.wts_global)
        self.res_global.append(abs(float(y_true) - float(y_hat)))
        self.wts_global.append(1.0)

        if self.by_regime:
            res_q, wts_q = self._buffers_for(regime_label)
            self._decay(wts_q)
            res_q.append(abs(float(y_true) - float(y_hat)))
            wts_q.append(1.0)

    def interval(
        self,
        y_hat: float,
        alpha: float = 0.1,
        regime_label: str | None = None,
        scale_hint: float | None = None,
        alphas_multi: list[float] | None = None,
    ):
        def _q_for(buf_res: deque[float], buf_wt: deque[float], a: float) -> float:
            if not buf_res:
                return float(scale_hint if scale_hint is not None else self.cold_scale)
            return _weighted_quantile(list(buf_res), list(buf_wt), 1 - a)

        res_q, wts_q = self._buffers_for(regime_label)

        if alphas_multi:
            out: dict[str, tuple[float, float]] = {}
            for a in alphas_multi:
                q = _q_for(res_q, wts_q, a)
                out[f"alpha={a:.2f}"] = (float(y_hat - q), float(y_hat + q))
            return out

        q = _q_for(res_q, wts_q, alpha)
        return float(y_hat - q), float(y_hat + q)
