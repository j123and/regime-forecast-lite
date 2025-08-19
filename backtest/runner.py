from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from backtest.metrics import coverage, latency_p50_p95, mae, rmse, smape


class BacktestRunner:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)

    def run(self, pipe, stream: Iterable[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
        log: list[dict[str, Any]] = []
        y_true_seq: list[float] = []
        y_pred_seq: list[float] = []
        ql_seq: list[float] = []
        qh_seq: list[float] = []
        lat_seq: list[float] = []

        prev_pred: dict[str, Any] | None = None

        for tick in stream:
            pred = pipe.process(tick)  # predicts y_{t+1|t}

            # score previous prediction against current truth y_t
            if prev_pred is not None:
                y_t = float(tick["x"])
                y_hat_prev = float(prev_pred["y_hat"])
                ql = float(prev_pred["interval_low"])
                qh = float(prev_pred["interval_high"])
                total_ms = float(prev_pred.get("latency_ms", {}).get("total", 0.0))

                y_true_seq.append(y_t)
                y_pred_seq.append(y_hat_prev)
                ql_seq.append(ql)
                qh_seq.append(qh)
                lat_seq.append(total_ms)

                log.append(
                    {
                        "t": tick["timestamp"],
                        "y_true": y_t,
                        "y_hat": y_hat_prev,
                        "ql": ql,
                        "qh": qh,
                        "regime": prev_pred.get("regime", ""),
                        "score": float(prev_pred.get("score", 0.0)),
                        "lat_ms": total_ms,
                    }
                )

                # critical: update conformal with residual for the SAME prediction we just scored
                pipe.update_truth(y_t, y_hat=y_hat_prev)

            prev_pred = pred

        m = {
            "mae": mae(y_true_seq, y_pred_seq),
            "rmse": rmse(y_true_seq, y_pred_seq),
            "smape": smape(y_true_seq, y_pred_seq),
            "coverage": coverage(y_true_seq, ql_seq, qh_seq),
        }
        p = latency_p50_p95(lat_seq)
        m["latency_p50_ms"] = p["p50"]
        m["latency_p95_ms"] = p["p95"]

        return m, log
