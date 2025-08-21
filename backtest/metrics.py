from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _pair_count(xs: Sequence[float], ys: Sequence[float]) -> int:
    """Number of elementwise pairs that will be compared (min lengths)."""
    return min(len(xs), len(ys))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Mean Absolute Error over available pairs. Returns NaN if no pairs.
    """
    n = _pair_count(y_true, y_pred)
    if n == 0:
        return float("nan")
    s = 0.0
    for a, b in zip(y_true, y_pred, strict=False):
        s += abs(a - b)
    return s / n


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Root Mean Squared Error over available pairs. Returns NaN if no pairs.
    """
    n = _pair_count(y_true, y_pred)
    if n == 0:
        return float("nan")
    ss = 0.0
    for a, b in zip(y_true, y_pred, strict=False):
        d = a - b
        ss += d * d
    return math.sqrt(ss / n)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Symmetric MAPE in percent. 0 if denominator is 0 for all pairs.
    """
    num = 0.0
    den = 0.0
    for a, f in zip(y_true, y_pred, strict=False):
        num += abs(f - a)
        den += (abs(a) + abs(f)) / 2.0
    return (num / den) * 100.0 if den > 0 else 0.0


def coverage(y_true: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    """
    Fraction of truths that lie within [lo, hi]. 0.0 if no pairs.
    """
    hit = 0
    total = 0
    for y, ql, qh in zip(y_true, lo, hi, strict=False):
        total += 1
        if ql <= y <= qh:
            hit += 1
    return hit / total if total else 0.0


def latency_p50_p95(latencies_ms: Sequence[float]) -> dict[str, float]:
    """
    p50 / p95 of latencies using simple order statistics. Returns zeros if empty.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    xs = sorted(latencies_ms)
    n = len(xs)
    p50 = xs[int(0.5 * (n - 1))]
    p95 = xs[int(0.95 * (n - 1))]
    return {"p50": float(p50), "p95": float(p95)}


def _indices_from_flags(flags: Sequence[int]) -> list[int]:
    return [i for i, f in enumerate(flags) if int(f) == 1]


def _match_events(
    true_idx: list[int], pred_idx: list[int], tol: int
) -> tuple[int, int, int, list[int]]:
    """
    Greedy bipartite matching with ±tol window.
    Returns (tp, fp, fn, delays_of_tp) where delay = pred_index - true_index (can be negative).
    """
    used_true: set[int] = set()
    tp = 0
    fp = 0
    delays: list[int] = []

    for p in pred_idx:
        cand: tuple[int, int] | None = None
        best_abs = tol + 1
        for t in true_idx:
            if t in used_true:
                continue
            d = p - t
            if -tol <= d <= tol and abs(d) < best_abs:
                best_abs = abs(d)
                cand = (t, d)
        if cand is None:
            fp += 1
        else:
            used_true.add(cand[0])
            tp += 1
            delays.append(cand[1])

    fn = len(true_idx) - len(used_true)
    return tp, fp, fn, delays


def _pred_indices_from_scores(
    scores: Sequence[float],
    threshold: float,
    cooldown: int,
    min_consecutive: int = 1,
) -> list[int]:
    """
    Threshold + debounce + cooldown.
    Fires at the first index of a run of >= min_consecutive scores above threshold,
    then enforces a cooldown gap of exactly `cooldown` subsequent indices.
    """
    preds: list[int] = []
    above = 0
    cool = 0
    for i, s in enumerate(scores):
        if cool > 0:
            cool -= 1
        if s >= threshold:
            above += 1
        else:
            above = 0
        if above >= min_consecutive and cool == 0:
            preds.append(i)
            cool = max(cooldown, 0)
            above = 0
    return preds


def detection_metrics(
    true_flags: Sequence[int] | None,
    scores: Sequence[float],
    threshold: float,
    cooldown: int,
    tol: int,
    min_consecutive: int = 1,
) -> dict[str, float]:
    """
    Core event detection metrics.
    """
    pred_idx = _pred_indices_from_scores(scores, threshold, cooldown, min_consecutive)
    pred_count = len(pred_idx)
    n = len(scores)
    chatter = (pred_count / max(n, 1)) * 1000.0

    out: dict[str, float] = {
        "cp_pred_count": float(pred_count),
        "cp_chatter_per_1000": float(chatter),
    }

    no_true = (not true_flags) or (sum(int(f) for f in true_flags) == 0)
    if no_true:
        far = pred_count / max(n, 1)
        out.update(
            {
                "cp_precision": float("nan"),
                "cp_recall": float("nan"),
                "cp_delay_mean": float("nan"),
                "cp_delay_p95": float("nan"),
                "cp_earliness_mean": float("nan"),
                "cp_false_alarm_rate": float(far),
            }
        )
        return out

    # mypy: after no_true is false, we know true_flags is not None
    assert true_flags is not None

    true_idx = _indices_from_flags(true_flags)
    tp, fp, fn, delays = _match_events(true_idx, pred_idx, tol)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    far = fp / max(n, 1)

    # separate tardiness (>0) and earliness (<0)
    tardy = [float(d) for d in delays if d > 0]
    early = [float(-d) for d in delays if d < 0]

    if tardy:
        t_sorted = sorted(tardy)
        delay_mean: float = sum(tardy) / len(tardy)
        delay_p95: float = t_sorted[int(0.95 * (len(t_sorted) - 1))]
    else:
        delay_mean = float("nan")
        delay_p95 = float("nan")

    out.update(
        {
            "cp_precision": float(prec),
            "cp_recall": float(rec),
            "cp_delay_mean": float(delay_mean),
            "cp_delay_p95": float(delay_p95),
            "cp_earliness_mean": float(sum(early) / len(early)) if early else float("nan"),
            "cp_false_alarm_rate": float(far),
        }
    )
    return out


def cp_event_metrics(
    log: list[dict[str, Any]], tol: int, *, threshold: float = 0.5, cooldown: int = 5
) -> dict[str, float]:
    """
    Thin wrapper to compute CP metrics from a backtest log.
    Expects 'cp_true' and 'score' in each log row.
    """
    true_flags = [int(row.get("cp_true", 0.0)) for row in log]
    scores = [float(row.get("score", 0.0)) for row in log]
    return detection_metrics(true_flags, scores, threshold=threshold, cooldown=cooldown, tol=tol)
