from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .metrics import coverage, latency_p50_p95, mae, rmse, smape


class BacktestRunner:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)

    def run(
        self,
        pipe,
        stream: Iterable[dict[str, Any]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        log: list[dict[str, Any]] = []
        y_true_seq: list[float] = []
        y_pred_seq: list[float] = []
        ql_seq: list[float] = []
        qh_seq: list[float] = []
        lat_seq: list[float] = []

        prev_tick: dict[str, Any] | None = None
        prev_pred: dict[str, Any] | None = None

        for tick in stream:
            out = pipe.process(tick)
            # finalize previous sample once we know current truth (1-step ahead)
            if prev_tick is not None and prev_pred is not None:
                y_true = float(prev_tick["x"])
                pipe.update_truth(y_true)
                y_true_seq.append(y_true)
                y_pred_seq.append(float(prev_pred["y_hat"]))
                ql_seq.append(float(prev_pred["interval_low"]))
                qh_seq.append(float(prev_pred["interval_high"]))
                lat_seq.append(float(prev_pred["latency_ms"]["total"]))
            log.append({**out, "timestamp": tick["timestamp"], "x": float(tick["x"])})
            prev_tick, prev_pred = tick, out

        metrics: dict[str, float] = {}
        if y_true_seq:
            metrics["mae"] = mae(y_true_seq, y_pred_seq)
            metrics["rmse"] = rmse(y_true_seq, y_pred_seq)
            metrics["smape"] = smape(y_true_seq, y_pred_seq)
            metrics["coverage"] = coverage(y_true_seq, ql_seq, qh_seq)
            lat = latency_p50_p95(lat_seq)
            metrics["latency_p50_ms"] = lat["p50"]
            metrics["latency_p95_ms"] = lat["p95"]
        return metrics, log
