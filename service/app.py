import time
from typing import Dict
import structlog
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from .schemas import PredictIn, PredictOut

log = structlog.get_logger()
app = FastAPI(title="Regime Forecast Service", version="0.1.0")

REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram("latency_ms", "Latency per stage (ms)", ["stage"], buckets=(1,2,5,10,20,50,100,200,500))

def timer(stage: str):
    start = time.perf_counter()
    def stop():
        ms = (time.perf_counter() - start) * 1000.0
        LAT.labels(stage).observe(ms)
        return ms
    return stop

def extract_features(x: float, cov: Dict[str, float]) -> Dict[str, float]:
    time.sleep(0)
    return {"z": x, **(cov or {})}

def detect_regime(feats: Dict[str, float]) -> tuple[str, float]:
    score = min(1.0, abs(feats.get("z", 0.0)) / 2.0)
    label = "high_vol" if score > 0.5 else "low_vol"
    return label, score

def route_model(regime: str) -> str:
    return "xgb" if regime == "high_vol" else "arima"

def forecast(model_name: str, x: float, feats: Dict[str, float]) -> float:
    return x

def conformal_interval(y_hat: float) -> tuple[float, float]:
    return y_hat - 0.01, y_hat + 0.01

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
    lat = {}

    stop = timer("extract")
    feats = extract_features(inp.x, inp.covariates or {})
    lat["extract"] = stop()

    stop = timer("detect")
    regime, score = detect_regime(feats)
    lat["detect"] = stop()

    stop = timer("route")
    model = route_model(regime)
    lat["route"] = stop()

    stop = timer("forecast")
    y_hat = forecast(model, inp.x, feats)
    lat["forecast"] = stop()

    stop = timer("conformal")
    ql, qh = conformal_interval(y_hat)
    lat["conformal"] = stop()

    lat["total"] = sum(lat.values())

    return PredictOut(
        y_hat=y_hat,
        interval_low=ql,
        interval_high=qh,
        regime=regime,
        score=score,
        latency_ms=lat,
    )
