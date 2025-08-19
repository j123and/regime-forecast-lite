from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .metrics import coverage, latency_p50_p95, mae, rmse, smape


class BacktestRunner:
    def __init__(self, alpha: float = 0.1, cp_tol: int = 10) -> None:
        # NOTE: alpha is kept for compatibility; wire it into interval construction in Pipeline if applicable.
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
        prev_latency: float = 0.0

        for tick in stream:
            if prev_tick is not None:
                # supply the last observed truth so the pipeline can update state
                pipe.update_truth(float(prev_tick["x"]))

            # get prediction for the current tick
            pred = pipe.process(tick)
            curr_latency = float(pred.get("latency_ms", {}).get("total_ms", 0.0))

            # metrics are computed as: previous prediction vs current truth
            if prev_pred is not None:
                y_true_seq.append(float(tick["x"]))
                y_pred_seq.append(float(prev_pred["y_hat"]))
                ql_seq.append(float(prev_pred["interval_low"]))
                qh_seq.append(float(prev_pred["interval_high"]))

                # CP truth for the current tick
                cp_true = float(tick.get("cp", tick.get("is_cp", 0.0)) or 0.0)

                # log the aligned pair using prev_pred and its latency
                log.append(
                    {
                        "t": tick["timestamp"],
                        "y": float(tick["x"]),
                        "y_hat": float(prev_pred["y_hat"]),
                        "ql": float(prev_pred["interval_low"]),
                        "qh": float(prev_pred["interval_high"]),
                        "regime": str(prev_pred.get("regime", "")),
                        "score": float(prev_pred.get("score", 0.0)),
                        "cp_prob": float(prev_pred.get("score", 0.0)),  # alias for now
                        "cp_true": cp_true,
                        "lat_total_ms": float(prev_latency),
                    }
                )

            # track latency series (for all produced preds)
            lat_seq.append(curr_latency)

            prev_tick = tick
            prev_pred = pred
            prev_latency = curr_latency

        m: dict[str, float] = {
            "mae": mae(y_true_seq, y_pred_seq),
            "rmse": rmse(y_true_seq, y_pred_seq),
            "smape": smape(y_true_seq, y_pred_seq),
            "coverage": coverage(y_true_seq, ql_seq, qh_seq),
        }
        p = latency_p50_p95(lat_seq)
        m["latency_p50_ms"] = p["p50"]
        m["latency_p95_ms"] = p["p95"]

        # Optional CP metrics if available
        try:
            from .metrics import cp_event_metrics

            cp_m: dict[str, float] = cp_event_metrics(log, tol=self.cp_tol)
            m.update(cp_m)
        except Exception:
            # Don't crash the run if CP metrics can't be computed.
            pass

        return m, log
