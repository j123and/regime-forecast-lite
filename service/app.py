from __future__ import annotations

import subprocess
import time

import structlog
from fastapi import FastAPI
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
pipe = Pipeline(cfg=cfg)

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

BUILD_INFO = {"sha": _git_sha(), "built_at": str(int(time.time()))}

@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/config")
def get_config() -> dict:
    return cfg

@app.get("/version")
def version() -> dict:
    return BUILD_INFO

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    tick: Tick = {"timestamp": inp.timestamp, "x": float(inp.x), "covariates": dict(inp.covariates or {})}
    out = pipe.process(tick)
    return PredictOut(**out)
