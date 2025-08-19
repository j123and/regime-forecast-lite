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
    Absolute-residual conformal with sliding window and exponential weighting.
    - window: max residuals stored
    - decay: per-update multiplier for existing weights (0<decay<=1). 1.0 => unweighted.
    - cold_scale: fallback radius when no residuals exist (e.g., use ewm_vol)
    """
    def __init__(self, window: int = 500, decay: float = 1.0, cold_scale: float = 0.0) -> None:
        self.window = int(window)
        self.decay = float(decay)
        self.cold_scale = float(cold_scale)
        self.res: deque[float] = deque(maxlen=self.window)
        self.wts: deque[float] = deque(maxlen=self.window)

    @property
    def n(self) -> int:
        return len(self.res)

    def update(self, y_hat: float, y_true: float) -> None:
        if self.decay < 1.0 and self.wts:
            self.wts = deque((w * self.decay for w in self.wts), maxlen=self.window)
        self.res.append(abs(float(y_true) - float(y_hat)))
        self.wts.append(1.0)

    def interval(
        self,
        y_hat: float,
        alpha: float = 0.1,
        scale_hint: float | None = None,
    ) -> tuple[float, float]:
        if not self.res:
            q = float(scale_hint if scale_hint is not None else self.cold_scale)
        else:
            n = len(self.res)
            q_alpha = max(0.0, min(1.0, 1.0 - alpha * (n + 1) / max(1, n)))
            q = _weighted_quantile(list(self.res), list(self.wts), q_alpha)
        y_hat = float(y_hat)
        return y_hat - q, y_hat + q
