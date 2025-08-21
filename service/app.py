# service/app.py  (LITE)
from __future__ import annotations

import time
import uuid
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from core.config import load_config
from core.pipeline import Pipeline
from core.types import Tick
from service.schemas import PredictIn, PredictOut, TruthIn, TruthOut

app = FastAPI()

# ---------------- Metrics (tiny) ----------------
REQS = Counter("requests_total", "Total requests", ["endpoint"])
SERVICE_LAT = Histogram(
    "request_service_ms",
    "End-to-end service latency (ms)",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)

# ---------------- Pipeline ----------------
cfg = load_config()
pipe = Pipeline(cfg)
_lock = Lock()  # stateful pipeline => guard access

# Idempotency keyed ONLY by prediction_id (no series_id/target_ts path)
_APPLIED_PRED_IDS: set[str] = set()


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    # expose Prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    """
    One-step-ahead prediction on a streaming tick.
    Returns a new prediction_id; truth must reference that id later.
    """
    REQS.labels("predict").inc()
    t0 = time.perf_counter()

    cov = {k: _to_float(v) for k, v in (inp.covariates or {}).items()}
    series_id = (inp.series_id or "default").strip() or "default"
    target_ts = inp.target_timestamp or inp.timestamp  # next-tick alignment by default

    tick: Tick = {"timestamp": inp.timestamp, "x": _to_float(inp.x), "covariates": cov}

    with _lock:
        pred = pipe.process(tick)

    pred_id = str(uuid.uuid4())
    with _lock:
        # register pending so /truth can update conformal buffers by id
        pipe.register_prediction(
            pred_id, float(pred.get("y_hat", 0.0)), str(pred.get("regime", ""))
        )

    t1 = time.perf_counter()
    service_ms = (t1 - t0) * 1000.0
    SERVICE_LAT.observe(service_ms)

    # Build response; keep fields but default safely
    return PredictOut(
        prediction_id=pred_id,
        series_id=series_id,
        target_timestamp=target_ts,
        y_hat=_to_float(pred.get("y_hat")),
        interval_low=_to_float(pred.get("interval_low")),
        interval_high=_to_float(pred.get("interval_high")),
        intervals=(pred.get("intervals") if isinstance(pred.get("intervals"), dict) else None),
        regime=str(pred.get("regime", "")),
        score=_to_float(pred.get("score")),
        latency_ms={"service_ms": service_ms},
        warmup=bool(pred.get("warmup", False)),
        degraded=bool(pred.get("degraded", False)),
    )


@app.post("/truth", response_model=TruthOut)
def truth(payload: TruthIn) -> TruthOut:
    """
    Idempotent truth ingestion keyed by prediction_id ONLY.
    """
    REQS.labels("truth").inc()

    if not payload.prediction_id:
        raise HTTPException(status_code=422, detail="Provide prediction_id.")

    # accept y under y / y_true / value
    y_val = None
    for k in ("y", "y_true", "value"):
        v = getattr(payload, k)
        if v is not None:
            y_val = _to_float(v)
            break
    if y_val is None:
        raise HTTPException(status_code=422, detail="Missing y/y_true/value in body.")

    pid = payload.prediction_id
    with _lock:
        if pid in _APPLIED_PRED_IDS:
            return TruthOut(status="ok", matched_by="prediction_id", idempotent=True)
        applied = pipe.update_truth_by_id(pid, float(y_val))
        if not applied:
            # unknown/expired id
            raise HTTPException(status_code=404, detail="Unknown prediction_id (not pending).")
        _APPLIED_PRED_IDS.add(pid)
        return TruthOut(status="ok", matched_by="prediction_id", idempotent=False)
