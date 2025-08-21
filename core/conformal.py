# core/conformal.py
from __future__ import annotations

from collections import deque


def _unweighted_quantile(vals: list[float], q: float) -> float:
    # linear interpolation between order stats
    n = len(vals)
    if n == 0:
        return 0.0
    if q <= 0.0:
        return float(min(vals))
    if q >= 1.0:
        return float(max(vals))
    s = sorted(vals)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _effective_n(wts: list[float]) -> float:
    s = sum(wts)
    s2 = sum(w * w for w in wts)
    return (s * s / s2) if s2 > 0.0 else 0.0


def _weighted_quantile(vals: list[float], wts: list[float], q: float) -> float:
    # vals already absolute residuals; wts >= 0
    assert 0.0 <= q <= 1.0
    if not vals:
        return 0.0
    pairs = sorted((float(v), float(w)) for v, w in zip(vals, wts, strict=False) if w > 0.0)
    total = sum(w for _, w in pairs)
    if total <= 0.0:
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
      - sliding window + optional exponential decay
      - optional per-regime buffers
      - multi-alpha interval computation
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
        if self.decay < 1.0:
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
        r = abs(float(y_true) - float(y_hat))
        self._decay(self.wts_global)
        self.res_global.append(r)
        self.wts_global.append(1.0)

        if self.by_regime:
            res_q, wts_q = self._buffers_for(regime_label)
            self._decay(wts_q)
            res_q.append(r)
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
            # empty buffer → cold scale or provided hint
            if not buf_res:
                base = scale_hint if scale_hint is not None else self.cold_scale
                return float(base)

            # guard: if effective N is small, use a safer unweighted 1−α quantile
            eff = _effective_n(list(buf_wt)) if buf_wt else 0.0
            if eff < 30.0:
                q_unw = _unweighted_quantile(list(buf_res), 1.0 - a)
                base = scale_hint if scale_hint is not None else self.cold_scale
                return float(max(q_unw, float(base)))

            # main: weighted 1−α quantile from the active buffer
            q_reg = _weighted_quantile(list(buf_res), list(buf_wt), 1.0 - a)

            # HARD GLOBAL FLOOR: never smaller than the global 1−α scale
            if self.res_global:
                q_glb = _unweighted_quantile(list(self.res_global), 1.0 - a)
                q_reg = max(q_reg, q_glb)
            return float(q_reg)

        res_q, wts_q = self._buffers_for(regime_label)

        if alphas_multi:
            out: dict[str, tuple[float, float]] = {}
            for a in alphas_multi:
                q = _q_for(res_q, wts_q, float(a))
                out[f"alpha={a:.2f}"] = (float(y_hat - q), float(y_hat + q))
            return out

        q = _q_for(res_q, wts_q, float(alpha))
        return float(y_hat - q), float(y_hat + q)
