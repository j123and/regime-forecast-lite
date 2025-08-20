# service/app.py
from __future__ import annotations

import atexit
import glob
import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from threading import Lock
from typing import Annotated, Any

import structlog
from fastapi import Body, FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from core.config import load_config
from core.pipeline import Pipeline
from core.types import Tick
from service.schemas import PredictIn, PredictOut, TruthIn, TruthOut

log = structlog.get_logger()
app = FastAPI()

# ---------------- Metrics ----------------
REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram(
    "latency_ms",
    "Latency per stage (ms)",
    ["stage"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)

# ---------------- Config / Pipeline ----------------
cfg = load_config()
pipe = Pipeline(cfg)
_lock = Lock()  # stateful pipeline => guard access

# ---------------- Auth (token) + Rate Limit ----------------
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "").strip()

RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "25"))     # tokens/sec
RATE_LIMIT_BURST = float(os.getenv("RATE_LIMIT_BURST", "50")) # bucket size
_RATE_BUCKETS: dict[str, tuple[float, float]] = {}            # key -> (tokens, last_ts)

def _auth_ok(req: Request) -> bool:
    if not SERVICE_TOKEN:
        return True
    return req.headers.get("authorization", "") == f"Bearer {SERVICE_TOKEN}"

def _rate_key(req: Request) -> str:
    tok = req.headers.get("authorization")
    if tok:
        return f"tok:{tok}"
    host = (req.client.host if req.client else "unknown") or "unknown"
    return f"ip:{host}"

def _allow_rate(key: str) -> bool:
    now = time.time()
    tokens, last = _RATE_BUCKETS.get(key, (RATE_LIMIT_BURST, now))
    delta = max(0.0, now - last)
    tokens = min(RATE_LIMIT_BURST, tokens + RATE_LIMIT_RPS * delta)
    if tokens < 1.0:
        _RATE_BUCKETS[key] = (tokens, now)
        return False
    _RATE_BUCKETS[key] = (tokens - 1.0, now)
    return True

@app.middleware("http")
async def _guard(req: Request, call_next):
    if not _auth_ok(req):
        REQS.labels("unauthorized").inc()
        return Response('{"detail":"unauthorized"}', status_code=401, media_type="application/json")
    if req.url.path in ("/predict", "/truth"):
        key = _rate_key(req)
        if not _allow_rate(key):
            REQS.labels("rate_limited").inc()
            return Response('{"detail":"rate limit"}', status_code=429, media_type="application/json")
    return await call_next(req)

# ---------------- Keying & Idempotency ----------------
# We now fully rely on Pipeline's pending map. The app only tracks mappings/early truths/idempotency.
SERIES_TS_TO_PRED_ID: dict[tuple[str, str], str] = {}  # (series_id, target_ts) -> pred_id
PRED_ID_TO_SERIES_TS: dict[str, tuple[str, str]] = {}  # pred_id -> (series_id, target_ts)
TRUTHS_EARLY: dict[tuple[str, str], float] = {}        # truths that arrived before predict
APPLIED_PRED_IDS: set[str] = set()
APPLIED_SERIES_TS: set[tuple[str, str]] = set()

# ---------------- Snapshots (persistence) ----------------
STATE_DIR = os.getenv("STATE_DIR", "state")
STATE_KEEP = int(os.getenv("STATE_KEEP", "5"))
os.makedirs(STATE_DIR, exist_ok=True)

def _snapshot_name() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"snapshot-{ts}.json"

def _list_snapshots() -> list[str]:
    return sorted(glob.glob(os.path.join(STATE_DIR, "snapshot-*.json")))

def _rotate_snapshots() -> None:
    files = _list_snapshots()
    if len(files) > STATE_KEEP:
        for f in files[: len(files) - STATE_KEEP]:
            try:
                os.remove(f)
            except Exception:
                pass

def _save_snapshot() -> str:
    snap = pipe.snapshot()
    path = os.path.join(STATE_DIR, _snapshot_name())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(snap, fh)
    os.replace(tmp, path)
    with open(os.path.join(STATE_DIR, "latest.json"), "w", encoding="utf-8") as fh:
        json.dump(snap, fh)
    _rotate_snapshots()
    return path

def _restore_from_path(path: str) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        snap = json.load(fh)
    with _lock:
        pipe.restore(snap)

def _restore_latest_if_present() -> str | None:
    files = _list_snapshots()
    path: str | None = files[-1] if files else None
    if not path:
        path = os.path.join(STATE_DIR, "latest.json")
        if not os.path.exists(path):
            return None
    try:
        _restore_from_path(path)
        log.info("state_restored", path=path)
        return path
    except Exception as e:
        log.warning("state_restore_failed", err=str(e))
        return None

# restore on boot (best-effort)
_restored_path = _restore_latest_if_present()

@atexit.register
def _persist_on_exit() -> None:
    try:
        p = _save_snapshot()
        log.info("state_snapshot_saved", path=p)
    except Exception as e:
        log.warning("state_snapshot_failed", err=str(e))

# ---------------- Helpers ----------------
def _to_float(v: Any, default: float = 0.0) -> float:
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
            pass

# ---------------- Endpoints ----------------
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

@app.get("/snapshots")
def list_snapshots() -> dict[str, Any]:
    REQS.labels("snapshots_list").inc()
    files = _list_snapshots()
    return {"files": [os.path.basename(f) for f in files], "latest": os.path.exists(os.path.join(STATE_DIR, "latest.json"))}

@app.post("/snapshot")
def snapshot_now() -> dict[str, Any]:
    REQS.labels("snapshot").inc()
    try:
        path = _save_snapshot()
        return {"status": "ok", "path": os.path.basename(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"snapshot_failed: {e}")

@app.post("/restore")
def restore_now(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    REQS.labels("restore").inc()
    name = None
    if isinstance(body, dict):
        name = body.get("name")
        latest = body.get("latest")
        if latest:
            name = None
    try:
        if name:
            path = os.path.join(STATE_DIR, os.path.basename(name))
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="snapshot_not_found")
            _restore_from_path(path)
            return {"status": "ok", "restored": os.path.basename(path)}
        else:
            restored = _restore_latest_if_present()
            if not restored:
                raise HTTPException(status_code=404, detail="no_snapshot_available")
            return {"status": "ok", "restored": os.path.basename(restored)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"restore_failed: {e}")

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn) -> PredictOut:
    REQS.labels("predict").inc()
    t0 = time.perf_counter()

    cov = {k: _to_float(v) for k, v in (inp.covariates or {}).items()}
    series_id = (inp.series_id or "default").strip() or "default"
    target_ts = inp.target_timestamp or inp.timestamp  # client should set explicit horizon

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

    # register with pipeline (multi-pending support)
    pred_id = str(uuid.uuid4())
    with _lock:
        pipe.register_prediction(pred_id, float(pred.get("y_hat", 0.0)), str(pred.get("regime", "")))
        SERIES_TS_TO_PRED_ID[(series_id, target_ts)] = pred_id
        PRED_ID_TO_SERIES_TS[pred_id] = (series_id, target_ts)
        # if truth arrived early, apply it now
        key = (series_id, target_ts)
        early = TRUTHS_EARLY.pop(key, None)
        if early is not None and key not in APPLIED_SERIES_TS:
            if pipe.update_truth_by_id(pred_id, float(early)):
                APPLIED_SERIES_TS.add(key)
                APPLIED_PRED_IDS.add(pred_id)

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
        prediction_id=pred_id,
        series_id=series_id,
        target_timestamp=target_ts,
    )
    return out

@app.post("/truth", response_model=TruthOut)
def truth(payload: TruthIn) -> TruthOut:
    """
    Idempotent keyed truth ingestion.
    Must include either prediction_id, or (series_id AND target_timestamp).
    """
    REQS.labels("truth").inc()

    # parse y
    y_val: float | None = None
    for k in ("y", "y_true", "value"):
        v = getattr(payload, k)
        if v is not None:
            y_val = _to_float(v)
            break
    if y_val is None:
        raise HTTPException(status_code=422, detail="Missing y/y_true/value in body.")

    pred_id = payload.prediction_id
    series_id = payload.series_id
    target_ts = payload.target_timestamp

    # Case 1: by prediction_id (preferred)
    if pred_id:
        with _lock:
            if pred_id in APPLIED_PRED_IDS:
                return TruthOut(status="ok", matched_by="prediction_id", idempotent=True)
            applied = pipe.update_truth_by_id(pred_id, float(y_val))
            if not applied:
                # unknown/expired id
                raise HTTPException(status_code=404, detail="Unknown prediction_id (not pending).")
            APPLIED_PRED_IDS.add(pred_id)
            st = PRED_ID_TO_SERIES_TS.get(pred_id)
            if st:
                APPLIED_SERIES_TS.add(st)
            return TruthOut(status="ok", matched_by="prediction_id", idempotent=False)

    # Case 2: by (series_id, target_timestamp)
    if not series_id or not target_ts:
        raise HTTPException(status_code=422, detail="Provide prediction_id OR (series_id AND target_timestamp).")

    key = (series_id, target_ts)
    with _lock:
        if key in APPLIED_SERIES_TS:
            return TruthOut(status="ok", matched_by="series_ts", idempotent=True)
        pid = SERIES_TS_TO_PRED_ID.get(key)
        if pid:
            applied = pipe.update_truth_by_id(pid, float(y_val))
            if applied:
                APPLIED_SERIES_TS.add(key)
                APPLIED_PRED_IDS.add(pid)
                return TruthOut(status="ok", matched_by="series_ts", idempotent=False)
            # mapping existed but pipeline no longer has it; treat as 404
            raise HTTPException(status_code=404, detail="Prediction for (series_id,target_timestamp) not pending.")
        # out-of-order: queue until predict arrives
        TRUTHS_EARLY[key] = float(y_val)
        return TruthOut(status="queued", matched_by="series_ts", queued=True)
