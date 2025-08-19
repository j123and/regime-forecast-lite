from __future__ import annotations

import math
from collections.abc import Sequence


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=False)) / max(1, len(y_true))

def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False)) / max(1, len(y_true))
    )

def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for a, f in zip(y_true, y_pred, strict=False):
        num += abs(f - a)
        s = abs(a) + abs(f)
        den += s if s != 0.0 else 1.0
    return 200.0 * num / den if den > 0 else 0.0

def coverage(y_true: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    hit = 0
    total = 0
    for y, ql, qh in zip(y_true, lo, hi, strict=False):
        total += 1
        if ql <= y <= qh:
            hit += 1
    return hit / total if total else 0.0

def latency_p50_p95(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    arr = sorted(latencies_ms)

    def pct(p: float) -> float:
        k = (len(arr) - 1) * p
        f = math.floor(k)
        c = min(f + 1, len(arr) - 1)
        if f == c:
            return arr[int(k)]
        return arr[f] + (arr[c] - arr[f]) * (k - f)

    return {"p50": pct(0.5), "p95": pct(0.95)}
