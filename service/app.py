import time
from typing import Dict
import structlog
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .schemas import PredictIn, PredictOut
from core.pipeline import Pipeline

log = structlog.get_logger()
app = FastAPI(title="Regime Forecast Service", version="0.2.0")
pipe = Pipeline()

REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram("latency_ms", "Latency per stage (ms)", ["stage"], buckets=(1,2,5,10,20,50,100,200,500))

def timer(stage: str):
    start = time.perf_counter()
    def stop():
        ms = (time.perf_counter() - start) * 1000.0
        LAT.labels(stage).observe(ms)
        return ms
    return stop

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    REQS.labels("healthz").inc()
    return {"status": "ok"}

@app.get("/metrics")
def metrics() -> Response:
    REQS.labels("metrics").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    lat: Dict[str, float] = {}

    tick = {"timestamp": inp.timestamp, "x": inp.x, "covariates": inp.covariates or {}}

    stop = timer("extract")
    feats = pipe.fe.update(tick)
    lat["extract"] = stop()

    stop = timer("detect")
    det = pipe.det.update(inp.x, feats)
    lat["detect"] = stop()

    stop = timer("route")
    model_name = pipe.router.choose(det["regime_label"], det["regime_score"])
    lat["route"] = stop()

    stop = timer("forecast")
    y_hat, _ = pipe.models[model_name].predict_update(tick, feats)
    lat["forecast"] = stop()

    stop = timer("conformal")
    ql, qh = pipe.conf.interval(y_hat, alpha=0.1)
    lat["conformal"] = stop()

    lat["total"] = sum(lat.values())

    return PredictOut(
        y_hat=float(y_hat),
        interval_low=float(ql),
        interval_high=float(qh),
        regime=str(det["regime_label"]),
        score=float(det["regime_score"]),
        latency_ms=lat,
    )
