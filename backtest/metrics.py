from __future__ import annotations

import math
from collections.abc import Sequence


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=False)) / n

def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    s = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False))
    return math.sqrt(s / n)

def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    eps = 1e-12
    num = 0.0
    for a, f in zip(y_true, y_pred, strict=False):
        num += abs(f - a) / max(abs(a) + abs(f) + eps, eps)
    return 2.0 * num / n

def coverage(y_true: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    total = 0
    hit = 0
    for y, ql, qh in zip(y_true, lo, hi, strict=False):
        total += 1
        if ql <= y <= qh:
            hit += 1
    return hit / total if total else 0.0

def latency_p50_p95(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    xs = sorted(float(x) for x in latencies_ms)
    def _pct(p: float) -> float:
        if not xs:
            return 0.0
        i = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[i]
    return {"p50": _pct(50.0), "p95": _pct(95.0)}
