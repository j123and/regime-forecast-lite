from collections import deque
import math
from typing import Deque, List, Tuple

def _weighted_quantile(vals: List[float], wts: List[float], q: float) -> float:
    assert 0.0 <= q <= 1.0
    if not vals:
        return 0.0
    pairs = sorted((abs(v), float(w)) for v, w in zip(vals, wts) if w > 0)
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
    - decay: multiplier applied to existing weights each update (e.g., 0.99); 1.0 = unweighted
    """
    def __init__(self, window: int = 500, decay: float = 1.0) -> None:
        self.window = int(window)
        self.decay = float(decay)
        self.res: Deque[float] = deque(maxlen=self.window)
        self.wts: Deque[float] = deque(maxlen=self.window)

    def update(self, y_hat: float, y_true: float) -> None:
        # decay existing weights
        if self.decay < 1.0 and self.wts:
            for i in range(len(self.wts)):
                self.wts[i] *= self.decay  # type: ignore[index]
        # append new residual with weight 1.0
        self.res.append(abs(float(y_true) - float(y_hat)))
        self.wts.append(1.0)

    def interval(self, y_hat: float, alpha: float = 0.1) -> Tuple[float, float]:
        if not self.res:
            q = 0.01  # cold start
        else:
            q = _weighted_quantile(list(self.res), list(self.wts), 1 - alpha)
        return float(y_hat - q), float(y_hat + q)
