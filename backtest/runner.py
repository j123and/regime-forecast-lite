from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .metrics import coverage, latency_p50_p95, mae, rmse, smape


class BacktestRunner:
    def __init__(self, alpha: float = 0.1, cp_tol: int = 10) -> None:
        self.alpha = float(alpha)
        self.cp_tol = int(cp_tol)

    def run(self, pipe, stream: Iterable[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        log: list[dict[str, Any]] = []

        y_true_seq: list[float] = []
        y_pred_seq: list[float] = []
        ql_seq: list[float] = []
        qh_seq: list[float] = []
        lat_seq: list[float] = []

        prev_tick: dict[str, Any] | None = None
        prev_pred: dict[str, Any] | None = None

        for tick in stream:
            if prev_tick is not None:
                pipe.update_truth(float(prev_tick["x"]))

            pred = pipe.process(tick)
            lat_seq.append(float(pred.get("latency_ms", {}).get("total_ms", 0.0)))

            if prev_pred is not None:
                y_true_seq.append(float(tick["x"]))
                y_pred_seq.append(float(prev_pred["y_hat"]))
                ql_seq.append(float(prev_pred["interval_low"]))
                qh_seq.append(float(prev_pred["interval_high"]))

            prev_tick = tick
            prev_pred = pred

            cp_true = float(tick.get("cp", tick.get("is_cp", 0.0)) or 0.0)

            log.append(
                {
                    "t": tick["timestamp"],
                    "y": float(tick["x"]),
                    "y_hat": float(pred["y_hat"]),
                    "ql": float(pred["interval_low"]),
                    "qh": float(pred["interval_high"]),
                    "regime": str(pred["regime"]),
                    "score": float(pred["score"]),
                    "cp_prob": float(pred.get("score", 0.0)),
                    "cp_true": cp_true,
                    "lat_total_ms": float(lat_seq[-1]),
                }
            )

        m: dict[str, float] = {
            "mae": mae(y_true_seq, y_pred_seq),
            "rmse": rmse(y_true_seq, y_pred_seq),
            "smape": smape(y_true_seq, y_pred_seq),
            "coverage": coverage(y_true_seq, ql_seq, qh_seq),
        }
        p = latency_p50_p95(lat_seq)
        m["latency_p50_ms"] = p["p50"]
        m["latency_p95_ms"] = p["p95"]

        # Optional CP metrics
        try:
            from .metrics import cp_event_metrics

            cp_m: dict[str, float] = cp_event_metrics(log, tol=self.cp_tol)
            m.update(cp_m)
        except Exception:
            pass

        return m, log
