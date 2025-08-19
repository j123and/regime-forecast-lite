from __future__ import annotations

import structlog
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from core.config import load_config
from core.pipeline import Pipeline
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

@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    tick = {"timestamp": inp.timestamp, "x": inp.x, "covariates": inp.covariates or {}}
    out = pipe.process(tick)  # type: ignore[arg-type]
    return PredictOut(**out)  # pydantic validates
