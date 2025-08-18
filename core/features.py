from collections import deque
from typing import Deque
import math
from .types import Tick, Features

_EPS = 1e-12

class FeatureExtractor:
    """
    Streaming features (no leakage):
      - rolling mean/std (window=win)
      - z-score (x - mean)/std
      - EWMA volatility on diffs (alpha)
      - lag-1 autocorrelation (window=win)
      - realized volatility: sqrt(mean of squared diffs over rv_win)
    """
    def __init__(self, win: int = 50, rv_win: int = 50, ewm_alpha: float = 0.1) -> None:
        self.win = int(win)
        self.rv_win = int(rv_win)
        self.ewm_alpha = float(ewm_alpha)

        self.buf: Deque[float] = deque(maxlen=self.win)
        self.sum = 0.0
        self.sumsq = 0.0

        self.prev_x: float | None = None
        self.rv_buf: Deque[float] = deque(maxlen=self.rv_win)
        self.rv_sum = 0.0

        self.ewm_var: float | None = None

    # --- helpers ---
    def _push_x(self, x: float) -> None:
        if len(self.buf) == self.win:
            old = self.buf.popleft()
            self.sum -= old
            self.sumsq -= old * old
        self.buf.append(x)
        self.sum += x
        self.sumsq += x * x

    def _rolling_mean_std(self) -> tuple[float, float]:
        n = len(self.buf)
        if n == 0:
            return 0.0, 0.0
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)
        std = math.sqrt(var)
        return mean, std

    def _update_rv(self, x: float) -> float:
        if self.prev_x is None:
            return 0.0
        d = x - self.prev_x
        s2 = d * d
        if len(self.rv_buf) == self.rv_win:
            self.rv_sum -= self.rv_buf.popleft()
        self.rv_buf.append(s2)
        self.rv_sum += s2
        n = len(self.rv_buf)
        return math.sqrt(self.rv_sum / n) if n > 0 else 0.0

    def _update_ewm_vol(self, x: float) -> float:
        if self.prev_x is None:
            if self.ewm_var is None:
                self.ewm_var = 0.0
            return 0.0
        alpha = self.ewm_alpha
        d = x - self.prev_x
        inst_var = d * d
        self.ewm_var = (1.0 - alpha) * (self.ewm_var or 0.0) + alpha * inst_var
        return math.sqrt(max(self.ewm_var, 0.0))

    def _ac1(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        xs = list(self.buf)
        mean = sum(xs) / n
        num = sum((xs[i] - mean) * (xs[i - 1] - mean) for i in range(1, n))
        den = sum((v - mean) ** 2 for v in xs)
        return float(num / den) if den > _EPS else 0.0

    # --- main update ---
    def update(self, tick: Tick) -> Features:
        x = float(tick["x"])

        # rolling stats
        self._push_x(x)
        mean, std = self._rolling_mean_std()
        z = (x - mean) / std if std > _EPS else 0.0

        # vols & ac1
        rv = self._update_rv(x)
        ewm_vol = self._update_ewm_vol(x)
        ac1 = self._ac1()

        # merge covariates last
        feats: Features = {"z": float(z), "ewm_vol": float(ewm_vol), "ac1": float(ac1), "rv": float(rv)}
        feats.update(tick.get("covariates", {}))

        self.prev_x = x
        return feats
