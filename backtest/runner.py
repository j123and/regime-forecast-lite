from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import time

from .metrics import coverage, latency_p50_p95, mae, rmse, smape


def _ingest_truth(pipe, y: float, prediction_id: str | None = None):
    """Feed realized truth into whatever method the pipeline exposes."""
    if hasattr(pipe, "update_truth"):
        try:
            return pipe.update_truth(y=y, prediction_id=prediction_id)
        except TypeError:
            return pipe.update_truth(y)
    for name in ("update", "learn_one", "observe_truth", "on_truth"):
        if hasattr(pipe, name):
            return getattr(pipe, name)(y)
    # No explicit truth method → assume pipeline self-updates in process()
    return None


def _predict(pipe, tick: dict[str, Any]) -> dict[str, Any]:
    """Call the pipeline's predict/step/process function."""
    for name in ("process", "predict", "step", "process_tick", "__call__"):
        if hasattr(pipe, name):
            fn = getattr(pipe, name)
            return fn(tick)
    raise AttributeError(
        "Pipeline has no predict method; expected one of: "
        "process/predict/step/process_tick/__call__"
    )


def _extract_latency_ms(pred: dict[str, Any]) -> float:
    lm = pred.get("latency_ms") or {}
    if isinstance(lm, dict):
        # Prefer total_ms; else service_ms; else compute_ms if someone set it
        for k in ("total_ms", "service_ms", "compute_ms"):
            if k in lm:
                try:
                    return float(lm[k])
                except Exception:
                    pass
    try:
        return float(lm)  # if someone returns a bare float
    except Exception:
        return 0.0


def _extract_yhat(pred: dict[str, Any]) -> float:
    for k in ("y_hat", "yhat", "y_pred", "prediction", "y"):
        if k in pred:
            return float(pred[k])
    raise KeyError("Prediction dict missing y_hat/yhat/y_pred/prediction/y")


def _extract_intervals(pred: dict[str, Any], alpha: float) -> tuple[float, float]:
    if "interval_low" in pred and "interval_high" in pred:
        return float(pred["interval_low"]), float(pred["interval_high"])
    iv = pred.get("intervals")
    if isinstance(iv, dict) and iv:
        key_variants = (f"alpha={alpha:.2f}", f"alpha={alpha}", alpha, str(alpha))
        for k in key_variants:
            if k in iv:
                low, high = iv[k]
                return float(low), float(high)
        first = next(iter(iv.values()))
        low, high = first
        return float(low), float(high)
    yhat = _extract_yhat(pred)
    return float(yhat), float(yhat)


class BacktestRunner:
    def __init__(
        self,
        alpha: float = 0.1,
        cp_tol: int = 10,
        *,
        cp_threshold: float | None = None,
        cp_cooldown: int | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.cp_tol = int(cp_tol)
        self.cp_threshold = cp_threshold
        self.cp_cooldown = cp_cooldown

    def run(
        self, pipe, stream: Iterable[dict[str, Any]]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
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
            # feed last tick's truth before predicting current tick
            if prev_tick is not None:
                _ingest_truth(pipe, float(prev_tick["x"]))

            # measure compute-time for this prediction
            t0 = time.perf_counter()
            pred = _predict(pipe, tick)
            t1 = time.perf_counter()
            compute_ms = (t1 - t0) * 1000.0

            # prefer latency reported by the model/service; else use compute time
            curr_latency = _extract_latency_ms(pred)
            if curr_latency == 0.0:
                curr_latency = compute_ms
                # expose it for anyone who reads the pred dict downstream
                pred.setdefault("latency_ms", {})  # type: ignore[arg-type]
                if isinstance(pred["latency_ms"], dict):  # type: ignore[index]
                    pred["latency_ms"]["compute_ms"] = compute_ms  # type: ignore[index]

            if prev_pred is not None:
                # Evaluate last prediction against current truth
                y_true_seq.append(float(tick["x"]))
                y_pred_seq.append(_extract_yhat(prev_pred))
                ql, qh = _extract_intervals(prev_pred, self.alpha)
                ql_seq.append(ql)
                qh_seq.append(qh)

                cp_true = float(tick.get("cp", tick.get("is_cp", 0.0)) or 0.0)

                log.append(
                    {
                        "t": tick.get("timestamp"),
                        "y": float(tick["x"]),
                        "y_hat": _extract_yhat(prev_pred),
                        "ql": ql,
                        "qh": qh,
                        "regime": str(prev_pred.get("regime", "")),
                        "score": float(prev_pred.get("score", 0.0)),
                        "cp_prob": float(prev_pred.get("score", 0.0)),
                        "cp_true": cp_true,
                        "lat_total_ms": float(prev_latency),
                    }
                )

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

            cp_kwargs: dict[str, float | int] = {}
            if self.cp_threshold is not None:
                cp_kwargs["threshold"] = float(self.cp_threshold)
            if self.cp_cooldown is not None:
                cp_kwargs["cooldown"] = int(self.cp_cooldown)

            cp_m: dict[str, float] = cp_event_metrics(log, tol=self.cp_tol, **cp_kwargs)  # type: ignore[arg-type]
            m.update(cp_m)
        except Exception:
            pass

        return m, log
