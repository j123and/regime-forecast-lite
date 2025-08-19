from __future__ import annotations

import functools
import subprocess
import time
from collections import defaultdict, deque

import structlog
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from core.config import load_config
from core.pipeline import Pipeline
from core.types import Tick
from service.schemas import FeedbackIn, FeedbackOut, PredictIn, PredictOut

log = structlog.get_logger()
app = FastAPI()

# Prometheus metrics
REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram("latency_ms", "Latency per stage (ms)", ["stage"], buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500))
COVER_TOTAL = Counter("conformal_total", "Conformal feedback count", ["alpha", "regime"])
COVER_HIT = Counter("conformal_hit", "Conformal coverage hits", ["alpha", "regime"])
COVER_ROLL = Gauge("conformal_coverage_rolling", "Rolling coverage (last N feedbacks)", ["alpha", "regime"])

# rolling coverage state
class _RollingCoverage:
    def __init__(self, window: int = 500) -> None:
        self.window = int(window)
        self._bufs: dict[tuple[str, str], deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, alpha: str, regime: str, hit: bool) -> float:
        key = (alpha, regime)
        d = self._bufs[key]
        d.append(1.0 if hit else 0.0)
        val = sum(d) / len(d)
        COVER_ROLL.labels(alpha=alpha, regime=regime).set(val)
        return val

cfg = load_config()
pipe = Pipeline(cfg)
ROLL = _RollingCoverage(window=int(cfg.get("metrics", {}).get("coverage_roll_window", 500)))

def timer(stage: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                LAT.labels(stage).observe((time.perf_counter() - t0) * 1e3)
        return wrapped
    return deco

@app.get("/healthz")
def healthz() -> dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}

@app.get("/version")
def version() -> dict[str, str]:
    REQS.labels("version").inc()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    return {"version": sha}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictOut)
@timer("predict")
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    tick: Tick = {"timestamp": inp.timestamp, "x": float(inp.x), "covariates": dict(inp.covariates)}
    out = pipe.process(tick)
    return PredictOut(**out)

@app.post("/feedback", response_model=FeedbackOut)
@timer("feedback")
def feedback(inp: FeedbackIn) -> FeedbackOut:
    """
    Provide the realized y_true for the PREVIOUS step.
    This updates conformal residuals and coverage metrics for that step.
    """
    REQS.labels("feedback").inc()
    res = pipe.feedback(inp.y_true)
    regime = str(res.get("regime", ""))
    hits = {str(a): bool(v) for a, v in (res.get("hits") or {}).items()}

    for a_str, hit in hits.items():
        COVER_TOTAL.labels(alpha=a_str, regime=regime).inc()
        if hit:
            COVER_HIT.labels(alpha=a_str, regime=regime).inc()
        ROLL.update(alpha=a_str, regime=regime, hit=hit)

    return FeedbackOut(regime=regime, hits=hits)
