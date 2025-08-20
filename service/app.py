from __future__ import annotations

import subprocess
import time
from threading import Lock
from typing import Any

import structlog
from fastapi import Body, FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from core.config import load_config
from core.pipeline import Pipeline
from core.types import Tick
from service.schemas import PredictIn, PredictOut

log = structlog.get_logger()
app = FastAPI()

REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram(
    "latency_ms",
    "Latency per stage (ms)",
    ["stage"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)

cfg = load_config()
pipe = Pipeline(cfg)
_lock = Lock()  # stateful pipeline => guard access


def _to_float(v: Any, default: float = 0.0) -> float:
    # accept ints/floats/strings that look like numbers
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return default
    return default


def _to_pair(v: Any) -> list[float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return [_to_float(v[0]), _to_float(v[1])]
    return None


def _observe_latencies(lat: dict[str, float]) -> None:
    for stage, ms in lat.items():
        try:
            LAT.labels(stage).observe(float(ms))
        except Exception:
            # never let metrics blow up the request path
            pass


@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}


@app.get("/config")
def config_echo() -> dict[str, Any]:
    REQS.labels("config").inc()
    return cfg


@app.get("/version")
def version() -> dict[str, str]:
    REQS.labels("version").inc()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        sha = "unknown"
    return {"git_sha": sha}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    t0 = time.perf_counter()

    cov = {k: _to_float(v) for k, v in (inp.covariates or {}).items()}
    tick: Tick = {"timestamp": inp.timestamp, "x": _to_float(inp.x), "covariates": cov}

    with _lock:
        pred = pipe.process(tick)

    t1 = time.perf_counter()

    # latencies
    lat: dict[str, float] = {}
    lat_raw = pred.get("latency_ms", {})
    if isinstance(lat_raw, dict):
        for k, v in lat_raw.items():
            if isinstance(v, (int, float)):
                lat[k] = float(v)
    lat["service_ms"] = (t1 - t0) * 1000.0
    _observe_latencies(lat)

    # optional multi-α intervals
    intervals_out: dict[str, list[float]] | None = None
    raw_intervals = pred.get("intervals")
    if isinstance(raw_intervals, dict):
        intervals_out = {}
        for k, pair in raw_intervals.items():
            p = _to_pair(pair)
            if p is not None:
                intervals_out[k] = p

    out = PredictOut(
        y_hat=_to_float(pred.get("y_hat")),
        interval_low=_to_float(pred.get("interval_low")),
        interval_high=_to_float(pred.get("interval_high")),
        regime=str(pred.get("regime", "")),
        score=_to_float(pred.get("score")),
        latency_ms=lat,
        warmup=bool(pred.get("warmup", False)),
        degraded=bool(pred.get("degraded", False)),
        intervals=intervals_out,
    )
    return out


@app.post("/truth")
def truth(payload: Any = Body(...)) -> dict[str, str]:
    """
    Accepts either a plain JSON number (e.g., 0.0005) or an object with 'y' or 'y_true'.
    """
    REQS.labels("truth").inc()

    y_val: float | None = None
    if isinstance(payload, (int, float, str)):
        y_val = _to_float(payload)
    elif isinstance(payload, dict):
        for k in ("y", "y_true", "value"):
            if k in payload:
                y_val = _to_float(payload[k])
                break

    if y_val is None:
        raise HTTPException(
            status_code=422,
            detail="Body must be a JSON number or an object containing 'y' or 'y_true'.",
        )

    with _lock:
        pipe.update_truth(y_val)
    return {"status": "ok"}
