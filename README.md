
# Regime Forecast

Online regime detection + adaptive forecasting with calibrated prediction intervals.
Built for streaming time series: detect change points, switch models on the fly, and serve next-tick forecasts with coverage you can trust.

## Highlights

* Online feature extraction (`z`, `ewm_vol`, `ac1`, `rv`) with no leakage
* Bayesian Online Change Point Detection (BOCPD) + score-aware router
* Two online forecasters out of the box:

  * SARIMAX (statsmodels) with sparse refits and fast append
  * Gradient boosting (XGBoost), falling back to `SGDRegressor`, then naïve if libs unavailable
* Online conformal intervals (per-regime buffers, decay, sliding window)
* FastAPI service with Prometheus metrics and OpenAPI docs
* Backtest CLI and hyperparameter sweep (materialized dataset = fast)
* Dockerfile, GitHub Actions CI, Grafana dashboard JSON, Prometheus alert rules
* Example dataset and minimal config to get you running quickly

---

## Quickstart

### 1) Dev install (Python 3.11)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2) Backtest on the tiny example

This is a toy; don’t expect calibrated coverage during warm-up.

```bash
python -m backtest.cli \
  --data examples/quickstart.csv \
  --config examples/config/minimal.yaml \
  --cp_tol 10 \
  --alpha 0.1 | python -m json.tool
```

### 3) Run the service once

```bash
uvicorn service.app:app --reload --port 8000 &
PID=$!; sleep 1

# health
curl -s http://localhost:8000/healthz | python -m json.tool

# one prediction
python - <<'PY' > /tmp/predict.json
import json
print(json.dumps({
  "timestamp":"2024-01-02T10:00:00Z",
  "x":0.001,
  "covariates":{"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
}))
PY
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' -d @/tmp/predict.json | python -m json.tool

# send truth (plain number is accepted)
echo 0.0007 | curl -s -X POST http://localhost:8000/truth -H 'content-type: application/json' --data-binary @- | python -m json.tool

# metrics (Prometheus)
curl -s http://localhost:8000/metrics | head -n 20

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

> Calibration note: with the `market` profile and α=0.1, coverage is \~0.90 after warm-up on AAPL 1h.

---

## What this does (plain English)

For each incoming tick `(timestamp, x, covariates)`:

1. **Features**: compute rolling stats (`z`, `ewm_vol`, `ac1`, `rv`) without peeking ahead.
2. **Change points**: BOCPD updates its posterior and yields a `regime_label` + continuous `regime_score`.
3. **Routing**: dwell/penalty/threshold policy picks a model (`arima` or `xgb`), with optional freeze after CP spikes.
4. **Forecast**: the chosen model outputs `y_hat` for the **next** tick and updates its own state online (no leakage).
5. **Intervals**: online conformal computes `[interval_low, interval_high]`. On the *next* tick, `/truth` updates residual buffers.

Backtesting aligns **previous prediction vs current truth**, and reports MAE/RMSE/sMAPE/coverage/latency plus change-point metrics.

---

## Requirements

* Python **3.11**
* Linux/macOS/WSL recommended (Dockerfile included)

---

## Data format

Timestamps must be UTC. Acceptable inputs:

* ISO-8601 with trailing `Z`, e.g., `2024-01-02T10:00:00Z`
* Unix epoch **seconds** as a string, e.g., `"1704180000"`

Timezone offsets (e.g., `+02:00`) are not supported.

Required columns:

* `timestamp` — ISO-8601 (UTC) or epoch string
* `x` — target float

Optional covariates (used by default):

* `rv`, `ewm_vol`, `ac1`, `z`

Optional labels:

* `cp` or `is_cp` — `0/1` CP indicator on the **current** tick

CSV and Parquet supported (Parquet requires `pyarrow`, already a dependency).

---

## Configuration

Resolution order:

1. `--config path.yaml`
2. `REGIME_CONFIG` env var
3. `--profile sim|market` or `REGIME_PROFILE` → `config/profiles/<profile>.yaml`
4. Fallback `config/default.yaml`

CLI `--alpha` **overrides** `conformal.alpha_main` and ensures it’s present in `conformal.alphas`.

Minimal example (also in `examples/config/minimal.yaml`):

```yaml
features:
  win: 50
  rv_win: 50
  ewm_alpha: 0.1

detector:
  threshold: 0.6
  cooldown: 5
  hazard: 0.005
  rmax: 400
  vol_threshold: 0.02

router:
  dwell_min: 10
  switch_threshold: 0.0
  switch_penalty: 0.0
  freeze_on_recent_cp: false
  freeze_ticks: 0

conformal:
  alpha_main: 0.1
  alphas: [0.1]
  window: 500
  decay: 1.0
  by_regime: true
  cold_scale: 0.01
```

---

## Backtesting & sweep

Single run:

```bash
python -m backtest.cli --data data/aapl_1h_logret.csv --profile market --cp_tol 10 --alpha 0.1
```

Fast sweep (dataset is read once; `--max_rows` limits workload):

```bash
python -m backtest.sweep --data data/aapl_1h_logret.csv --cp_tol 10 --alpha 0.1 --max_rows 5000 > sweep.jsonl
```

---

## API (FastAPI)

### Alignment & horizon

* One-step ahead only. Each `POST /predict` forecasts the **next tick** after the request `timestamp`.
* `/truth` currently applies to the **most recent outstanding prediction (FIFO)**. Out-of-order or duplicate truths are not supported and can corrupt calibration.

**Docs / UIs**: `/docs` (Swagger), `/redoc`
**Schema**: `/openapi.json` (see also `openapi/explicit-schemas.json` for the liberal `/truth` body)

**Endpoints**

`POST /predict` — request

```json
{
  "timestamp": "2024-01-02T10:00:00Z",
  "x": 0.001,
  "covariates": {"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
}
```

`POST /predict` — response

```json
{
  "y_hat": 0.0009,
  "interval_low": -0.0012,
  "interval_high": 0.0031,
  "intervals": {"alpha=0.10":[-0.0012,0.0031]},
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

`POST /truth` — body can be a number or object with `y`/`y_true`/`value`

```json
0.0007
{"y": 0.0007}
{"y_true": 0.0007}
{"value": 0.0007}
```

`GET /healthz` → `{"status":"ok"}`
`GET /version` → `{"git_sha":"<commit>"}`
`GET /config` → current loaded config (JSON)
`GET /metrics` → Prometheus exposition

---

---

## Benchmarks

Environment: WSL2, Python 3.11, `uvicorn` 1 worker (stateful), C=4, N=1000.
Method: `scripts/bench_light.sh` (curl+xargs), end-to-end HTTP.

* Requests OK: **1000 / 1000**
* Elapsed: **6.95 s**
* Throughput: **\~143.9 req/s**
* Latency: **p50 \~28.2 ms**, **p95 \~148.8 ms**
* RSS: *(print from script; varies by box)*

How to reproduce:

```bash
PORT=8000 N=1000 C=4 ./scripts/bench_light.sh
```

Notes:

* Single worker is deliberate (the service keeps state in-process).
* HTTP numbers include JSON encode/decode + FastAPI overhead. Internal pipeline timings are lower (see API `latency_ms` fields and the backtest metrics).

## Calibration

We report coverage for α=0.1 (90% PI), rolling coverage, and per-regime coverage.
α=0.1 achieves ~0.90 coverage after warm-up; p50/p95 pipeline latency ~8/10 ms on this box.
Quick JSON report:

```bash
python scripts/calibration_report.py data/aapl_1h_logret.csv
```

Example output (AAPL 1h, `market` profile):

```
{
  "out": "backtest_plot.png",
  "n_points_plotted": 1000,
  "metrics": {
    "mae": 0.005639353119261354,
    "rmse": 0.008414121952905976, #on AAPL 1h log returns
    "coverage": 0.9005291005291005,
    "latency_p50_ms": 7.876808999981222,
    "latency_p95_ms": 9.795716000098764,
    "cp_pred_count": 12.0,
    "cp_chatter_per_1000": 6.349206349206349,
    "cp_false_alarm_rate": 0.006349206349206349
  }
}
```

Plot (last 1000 points with intervals + regime shading):

```bash
python scripts/plot_backtest.py \
  --data data/aapl_1h_logret.csv \
  --profile market \
  --alpha 0.1 \
  --last 1000 \
  --out backtest_plot.png
```

Add two screenshots to this section:

* `backtest_plot.png` (truth vs ŷ, shaded regimes, bands)
* A cropped snippet of the JSON report above (or paste the JSON)

---

## Metrics, Grafana, Alerts

Prometheus:

* `requests_total{endpoint="predict|truth|healthz|version|config"}`
* `latency_ms_bucket{stage="features_ms|detector_ms|router_ms|model_ms|conformal_ms|service_ms"}` (+ `_sum`, `_count`)

Ready-to-import:

* Grafana dashboard: `grafana/dashboard.json`
* Prometheus alert rules: `prometheus/alerts.yaml`

---

## Docker

Build:

```bash
docker build -t regime-forecast:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 regime-forecast:latest
```

Config/data in container:

```bash
docker run --rm -p 8000:8000 \
  -e REGIME_PROFILE=market \
  -v "$PWD/config":/app/config \
  -v "$PWD/data":/app/data \
  regime-forecast:latest
```

---

## CI

GitHub Actions at `.github/workflows/ci.yml`:

* installs deps, runs `ruff` + `mypy` + `pytest`
* builds Docker image
* optional publish to GHCR on `main`

Optional: add calibration checks (coverage, rolling coverage, latency) to CI; see `scripts/ci_checks.sh` if you include it.

---

## Extending

* Add a model: implement `OnlineModel.predict_update(tick, feats)` and register it in `core/pipeline.Pipeline.models`.
* Add features: extend `FeatureExtractor`; avoid leakage; keep constant-time updates.
* Change routing: see `core/router.Router` (dwell/threshold/penalty/freeze).

---

## Known limitations

* **State is in-process**; multiple uvicorn workers do not share model/conformal state.
* If `statsmodels` / `xgboost` / `scikit-learn` aren’t available, models degrade to naïve but continue serving.
* Conformal uses absolute residuals; on near-zero series, sMAPE is volatile.

