from collections import deque
import math

class OnlineConformal:
    """Absolute-residual conformal with sliding window."""
    def __init__(self, window: int = 500) -> None:
        self.window = window
        self.resids = deque(maxlen=window)

    def update(self, y_hat: float, y_true: float) -> None:
        self.resids.append(abs(float(y_true) - float(y_hat)))

    def interval(self, y_hat: float, alpha: float = 0.1) -> tuple[float, float]:
        if not self.resids:
            q = 0.01  # cold-start width
        else:
            s = sorted(self.resids)
            k = min(len(s) - 1, max(0, int(math.ceil((1 - alpha) * (len(s) + 1)) - 1)))
            q = s[k]
        return float(y_hat - q), float(y_hat + q)
