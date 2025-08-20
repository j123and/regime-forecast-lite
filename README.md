

# Regime Forecast

Online regime detection + adaptive forecasting with calibrated prediction intervals.
Built for streaming time series: detect change points, switch models on the fly, and serve next-tick forecasts with coverage you can trust.

## Highlights

* Online feature extraction (`z`, `ewm_vol`, `ac1`, `rv`) with zero leakage.
* Bayesian Online Change Point Detection (BOCPD) + score-aware router.
* Two online forecasters out of the box:

  * SARIMAX (statsmodels) with sparse refits and fast append.
  * XGBoost; falls back to `SGDRegressor`; then naïve if libs unavailable.
* Online conformal intervals (sliding window + optional decay; optional per-regime buffers).
* FastAPI service with Prometheus metrics and OpenAPI docs.
* **Keyed, idempotent API**: every prediction has a `prediction_id`; `/truth` can reference it directly; duplicates are rejected.
* **Out-of-order truth**: accepted safely by `prediction_id` or `(series_id, target_timestamp)`.
* **Auth + rate limit**: Bearer token and per-token/IP rate limiting.
* **State snapshots**: snapshot/restore model + conformal buffers; simple file-based rotation.
* Backtest harness and fast sweep CLI (materialized dataset).
* Dockerfile, GitHub Actions CI, Grafana dashboard JSON, Prometheus alert rules.
* Example dataset + minimal config that gets you running in \~60 seconds.

---

## Quickstart

### 1) Install (Python 3.11)

Recommended: use the supplied constraints for reproducible builds.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# reproducible pins (generated from a working env)
pip install -r constraints.txt

# core + dev tools + service niceties (uvloop/httptools)
pip install -e ".[dev,service]"
```

### 2) Backtest on the tiny example

This is a toy; expect calibration to settle after warm-up.

```bash
python -m backtest.cli \
  --data examples/quickstart.csv \
  --config examples/config/minimal.yaml \
  --cp_tol 10 \
  --alpha 0.1 | python -m json.tool
```

### 3) Run the service and hit it once

```bash
# optional auth / rate limits (recommended)
export SERVICE_TOKEN='s3cr3t'
export RATE_LIMIT_RPS=1000
export RATE_LIMIT_BURST=2000

uvicorn service.app:app --port 8000 --workers 1 --loop uvloop --http httptools &
PID=$!; sleep 1

curl -s http://127.0.0.1:8000/healthz | python -m json.tool

# one prediction (keyed + target timestamp)
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $SERVICE_TOKEN" \
  -d '{
    "series_id":"demo",
    "timestamp":"2024-01-02T10:00:00Z",
    "target_timestamp":"2024-01-02T11:00:00Z",
    "x":0.001,
    "covariates": {"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
  }' | python -m json.tool

# metrics (Prometheus exposition)
curl -s http://127.0.0.1:8000/metrics | head -n 20

kill $PID
```

### Optional: use real data

```bash
python -m data.yahoo_fetch --ticker AAPL --start 2024-01-01 --end 2024-03-01 \
  --interval 1h --field logret --out data/aapl_1h_logret.csv

python -m backtest.cli \
  --data data/aapl_1h_logret.csv \
  --profile market \
  --cp_tol 10 \
  --alpha 0.1 | python -m json.tool
```

---

## What this does

For each incoming tick `(timestamp, x, covariates)`:

1. **Features**: compute rolling stats and covariates (`z`, `ewm_vol`, `ac1`, `rv`) with no peeking.
2. **Change points**: BOCPD updates its posterior and yields a `regime_label` plus a continuous `regime_score`.
3. **Routing**: dwell/penalty/threshold policy picks a model (`arima` or `xgb`), with optional freeze after CP spikes.
4. **Forecast**: chosen model produces `y_hat` for the **next tick** and updates internal state online (no leakage).
5. **Intervals**: online conformal computes `[interval_low, interval_high]`. When truth for that horizon arrives, residual buffers update.

Backtesting aligns **previous prediction vs current truth**, and reports MAE/RMSE/sMAPE/coverage/latency and change-point metrics.

---

## Requirements

* Python 3.11
* Linux/macOS/WSL recommended (Dockerfile included)

---

## Data format

Timestamps must be UTC.

Accepted:

* ISO-8601 with trailing `Z`, e.g. `2024-01-02T10:00:00Z`
* Unix epoch **seconds** as a string, e.g. `"1704180000"`

Required fields:

* `timestamp` — ISO-8601 `Z` or epoch string
* `x` — target float

Optional covariates:

* `rv`, `ewm_vol`, `ac1`, `z`

Optional labels:

* `cp` or `is_cp` — `0/1` CP indicator on the **current** tick

CSV works out of the box; Parquet needs `pyarrow` (already a dependency).

---

## Configuration

Load order:

1. `--config path.yaml`
2. `REGIME_CONFIG` environment variable
3. `--profile sim|market` or `REGIME_PROFILE` → `config/profiles/<profile>.yaml`
4. Fallback `config/default.yaml`

CLI `--alpha` **overrides** `conformal.alpha_main` and is injected into `conformal.alphas`.

A minimal example is in `examples/config/minimal.yaml`.

---

## API (FastAPI)

Docs: `/docs` and `/redoc`
Schema: `/openapi.json`

### Alignment & horizon

* Next-tick only (one-step ahead).
* Predictions are keyed; `/truth` is idempotent and accepts **out-of-order** submissions by `prediction_id` or `(series_id, target_timestamp)`.

### Endpoints

`POST /predict` — request

```json
{
  "series_id": "my_series",
  "timestamp": "2024-01-02T10:00:00Z",
  "target_timestamp": "2024-01-02T11:00:00Z",
  "x": 0.001,
  "covariates": {"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
}
```

`POST /predict` — response (example)

```json
{
  "prediction_id": "a9e9b9cc-4c4f-4d69-b3c0-0c6d2f7e5f0f",
  "y_hat": 0.0009,
  "interval_low": -0.0012,
  "interval_high": 0.0031,
  "intervals": {"alpha=0.10": [-0.0012, 0.0031]},
  "regime": "low_vol",
  "score": 0.42,
  "latency_ms": {
    "features_ms": 3.1,
    "detector_ms": 0.9,
    "router_ms": 0.1,
    "model_ms": 1.8,
    "conformal_ms": 0.3,
    "service_ms": 6.4
  },
  "warmup": false,
  "degraded": false
}
```

`POST /truth` — idempotent, out-of-order safe

Preferred (by `prediction_id`):

```json
{
  "prediction_id": "a9e9b9cc-4c4f-4d69-b3c0-0c6d2f7e5f0f",
  "y_true": 0.0007
}
```

Alternative (by `(series_id, target_timestamp)`):

```json
{
  "series_id": "my_series",
  "target_timestamp": "2024-01-02T11:00:00Z",
  "y_true": 0.0007
}
```

`/truth` replies with a small status document and dedup flag. Replays return HTTP 409.

Utility:

* `GET /healthz` → `{"status":"ok"}`
* `GET /version` → `{"git_sha":"<commit>" }`
* `GET /config` → current JSON config
* `GET /metrics` → Prometheus exposition
* `POST /snapshot` → saves `state/snapshot-YYYYMMDD-HHMMSS.json` and returns its path
* `POST /restore` → loads a snapshot path or the latest one

### Auth and rate limit

* Set `SERVICE_TOKEN` to enable Bearer token auth.
* Requests must carry `Authorization: Bearer <token>`.
* Rate limits per token/IP are controlled via `RATE_LIMIT_RPS` and `RATE_LIMIT_BURST`.
* Exceeding limits returns HTTP 429 with a JSON body including `retry_after_s`.

### Error responses (shape is stable)

```json
{ "error": "short_code", "detail": "human-readable cause", "status": 401 }
```

Examples:

* 401 Unauthorized — missing/invalid token.
* 403 Forbidden — token not permitted.
* 409 Conflict — duplicate `/truth` for the same `prediction_id` or `(series_id, target_timestamp)`.
* 422 Unprocessable Entity — bad types, missing fields, or bad timestamp format.
* 429 Too Many Requests — rate limit exceeded.

See `docs/errors.md` for concrete JSON for each case.

---

## Calibration evidence (real data)

Ran with the `market` profile and α=0.10 (90% PI).

AAPL 1h (last 1000 points):

* MAE 0.0056393
* RMSE 0.0084141
* Coverage 0.900529
* p50 pipeline latency \~7.63–7.88 ms
* p95 pipeline latency \~9.60–10.19 ms
* Artifact: `artifacts/calib/aapl_1h_logret_calibration.json`

MSFT 1h:

* Coverage 0.852632
* p50 \~4.27 ms
* p95 \~5.41 ms
* Artifact: `artifacts/calib/msft_1h_logret_calibration.json`

BTC-USD 1h (1000 points windowed):

* Coverage 0.917942
* p50 \~7.60 ms
* p95 \~11.47 ms
* Artifact: `artifacts/calib/btcusd_1h_logret_calibration.json`

Notes:

* Coverage stabilizes after warm-up; **no leakage** in conformal (intervals computed from pre-truth residual buffers).
* Per-regime coverage varies; high-volatility segments are expected to be tighter—monitor and alert on under-coverage.

How to reproduce the above:

```bash
python scripts/calibration_suite.py \
  --data data/aapl_1h_logret.csv data/msft_1h_logret.csv data/btcusd_1h_logret.csv \
  --profile market \
  --alphas 0.05,0.10,0.20 \
  --plot
# JSON summaries land in artifacts/calib/*.json; a combined summary at artifacts/calib/summary.jsonl
```

---

## Benchmarks (what you actually measured)

Environment: WSL2; Python 3.11; uvicorn with `uvloop` and `httptools`.
The service keeps state in-process; **one locked pipeline per worker** → requests serialize within a worker.

Single worker:

* c=1: \~99.12 rps; p50 ≈ 7.9 ms; p95 ≈ 9.9 ms; p99 ≈ 128.6 ms.
* c=4: \~89.17 rps; p50 ≈ 35.2 ms; p95 ≈ 155.2 ms; p99 ≈ 167.6 ms.

Four workers:

* c=1: \~96.11 rps; p50 ≈ 8.0 ms; p95 ≈ 10.9 ms; p99 ≈ 128.3 ms.
* c=16: \~264.17 rps; p50 ≈ 38.9 ms; p95 ≈ 162.5 ms; p99 ≈ 198.2 ms.

Reproduce quickly:

```bash
# put hey on PATH (e.g., /snap/bin)
export PATH="/snap/bin:$PATH"
export SERVICE_TOKEN='s3cr3t'
export RATE_LIMIT_RPS=100000
export RATE_LIMIT_BURST=100000

uvicorn service.app:app --port 8000 --workers 4 --loop uvloop --http httptools &

BODY='{"series_id":"bench","timestamp":"2024-01-02T10:00:00Z","target_timestamp":"2024-01-02T11:00:00Z","x":0.001,"covariates":{"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}}'

hey -n 4000 -c 1  -m POST -H 'content-type: application/json' -H "authorization: Bearer $SERVICE_TOKEN" -d "$BODY" http://127.0.0.1:8000/predict
hey -n 4000 -c 16 -m POST -H 'content-type: application/json' -H "authorization: Bearer $SERVICE_TOKEN" -d "$BODY" http://127.0.0.1:8000/predict
```

Interpretation:

* Throughput improves by adding **workers**.
* Within a worker, the pipeline lock serializes `/predict` → higher concurrency mostly increases queueing. If you need intra-worker scaling, move to per-series pipeline maps and per-series locks.

Artifacts:

* `artifacts/bench/summary.json` — parsed p50/p95/p99 and RPS.
* `artifacts/bench/env.json` — commit + Python/OS stamp.

---

## Metrics, Grafana, Alerts

Prometheus metrics:

* `requests_total{endpoint="predict|truth|healthz|version|config"}`
* `latency_ms_bucket{stage="features_ms|detector_ms|router_ms|model_ms|conformal_ms|service_ms"}` (+ `_sum`, `_count`)

Import:

* Grafana dashboard: `grafana/dashboard.json`
* Prometheus alert rules: `prometheus/alerts.yaml`

Recommended alerts:

* Under-coverage for α=0.1 below 0.85 over rolling window.
* Spike in `cp_chatter_per_1000`.
* Error rate or 429 rate spikes.

---

## Docker

Build:

```bash
docker build -t regime-forecast:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e SERVICE_TOKEN=s3cr3t \
  -e RATE_LIMIT_RPS=1000 \
  -e RATE_LIMIT_BURST=2000 \
  regime-forecast:latest
```

Mount config/data:

```bash
docker run --rm -p 8000:8000 \
  -e REGIME_PROFILE=market \
  -v "$PWD/config":/app/config \
  -v "$PWD/data":/app/data \
  regime-forecast:latest
```

---

## Packaging, versioning, changelog

* `pyproject.toml` defines the project; install with `pip install -e .` or with extras: `pip install -e ".[dev,service]"`.
* `constraints.txt` pins runtime versions; use `pip install -r constraints.txt` first for reproducibility.
* Versioning follows SemVer; see `CHANGELOG.md` for entries. The service exposes `/version` with the current git SHA.

---

## Extending

* Add a model: implement `OnlineModel.predict_update(tick, feats)` and register it in `core/pipeline.Pipeline.models`.
* Add features: extend `FeatureExtractor`; stick to constant-time updates; avoid leakage.
* Change routing: see `core/router.Router` (dwell/threshold/penalty/freeze).

---

## Known limitations

* **State is in-process**. Multiple uvicorn workers do not share model/conformal state. If you need shared state, use a single worker or back a state store (SQLite/Redis) and wire it into snapshot/restore points.
* Single locked pipeline per worker → requests serialize within a worker. For more throughput use multiple workers; for real intra-worker parallelism, move to per-series pipeline instances + per-series locks.
* Conformal uses absolute residuals; on near-zero series sMAPE is noisy.

---

## License

MIT. See `LICENSE`.

---

## Security

* Use `SERVICE_TOKEN` to require Bearer token auth.
* Rate limits via `RATE_LIMIT_RPS` and `RATE_LIMIT_BURST`.
* Don’t expose `/restore` or `/snapshot` publicly; protect them behind auth and network policy.

---

## Repro notes

Benchmark summaries live in `artifacts/bench/summary.json`.
Calibration summaries live in `artifacts/calib/*calibration.json` and `artifacts/calib/summary.jsonl`.
Each run stamps environment info at `artifacts/bench/env.json`.
