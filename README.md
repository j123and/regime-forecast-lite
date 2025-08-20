# Regime Forecast

Online regime detection + adaptive forecasting with calibrated prediction intervals.  
Built for streaming time-series: detect change points, switch models on the fly, and serve next-tick forecasts with coverage you can trust.

## Highlights

- Online feature extraction (`z`, `ewm_vol`, `ac1`, `rv`) with no leakage
- Bayesian Online Change Point Detection (BOCPD) + score-aware router
- Two online forecasters out of the box:
  - SARIMAX (statsmodels) with sparse refits and fast append
  - Gradient boosting (XGBoost), falling back to `SGDRegressor`, then naïve if libs unavailable
- Online conformal intervals (per-regime buffers, decay, sliding window)
- FastAPI service with Prometheus metrics and OpenAPI docs
- Backtest CLI and hyperparameter sweep (materialized dataset = fast)
- Dockerfile, GitHub Actions CI, Grafana dashboard JSON, Prometheus alert rules
- Example dataset and minimal config to get you running in 60 seconds

---

## Quickstart (60 seconds)

### 1) Dev install (Python 3.11)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
````

### 2) Backtest on the tiny example

```bash
python -m backtest.cli \
  --data examples/quickstart.csv \
  --config examples/config/minimal.yaml \
  --cp_tol 10 \
  --alpha 0.1 | python -m json.tool
```

### 3) Run the service and hit it once

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

# metrics (Prometheus exposition)
curl -s http://localhost:8000/metrics | head -n 20

kill $PID
```

### Optional: use real data

```bash
python -m data.yahoo_fetch --ticker AAPL --start 2024-01-01 --end 2024-03-01 --interval 1h --field logret --out data/aapl_1h_logret.csv

python -m backtest.cli \
  --data data/aapl_1h_logret.csv \
  --profile market \
  --cp_tol 10 \
  --alpha 0.1 | python -m json.tool
```

---

## What this does (in plain English)

For each incoming tick `(timestamp, x, covariates)`:

1. **Features**: compute rolling stats and covariates (`z`, `ewm_vol`, `ac1`, `rv`) without peeking ahead.
2. **Change points**: BOCPD updates its posterior and yields a `regime_label` plus a continuous `regime_score`.
3. **Routing**: a dwell/penalty/threshold policy picks a model (`arima` or `xgb`), with optional freeze after CP spikes.
4. **Forecast**: the chosen model produces `y_hat` for the next tick and updates its own state online (no leakage).
5. **Intervals**: online conformal computes `[interval_low, interval_high]`. As truth arrives on the *next* tick, residual buffers update.

Backtesting aligns **previous prediction vs current truth**, and reports MAE/RMSE/sMAPE/coverage/latency plus change-point metrics.

---

## Requirements

* Python **3.11**
* Linux/macOS/WSL recommended (Dockerfile included)

---

## Data format
Timestamps must be UTC. Acceptable inputs:
- ISO-8601 with trailing `Z`, e.g. `2024-01-02T10:00:00Z`
- Unix epoch **seconds** as a string, e.g. `"1704180000"`

Timezone offsets (e.g. `+02:00`) are not supported.

Required:

* `timestamp` — ISO 8601 string (UTC) or epoch string
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

CLI `--alpha` **overrides** `conformal.alpha_main` and ensures it’s in `conformal.alphas`.

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

- One-step ahead only. Each `POST /predict` produces a forecast for the **next tick** after the request `timestamp`.
- `/truth` currently applies to the **most recent outstanding prediction (FIFO)**. Out-of-order or duplicate truths are not supported and can corrupt calibration. This will change in a future version to require an explicit identifier.

* **Docs / UIs**: `/docs` (Swagger), `/redoc`
* **Schema**: `/openapi.json` (also see `openapi/explicit-schemas.json` for the liberal `/truth` body)

Endpoints:

* `POST /predict`
  Request body:

  ```json
  {
    "timestamp": "2024-01-02T10:00:00Z",
    "x": 0.001,
    "covariates": {"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
  }
  ```

  Response:

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

* `POST /truth`
  Body can be a number or object with `y`/`y_true`:

  0.0007
  {"y": 0.0007}
  {"y_true": 0.0007}
  {"value": 0.0007}


* `GET /healthz` → `{"status":"ok"}`

* `GET /version` → `{"git_sha":"<commit>" }`

* `GET /config` → current loaded config (as JSON)

* `GET /metrics` → Prometheus exposition

---

## Metrics, Grafana, Alerts

Prometheus counters/histograms:

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

GitHub Actions workflow at `.github/workflows/ci.yml`:

* installs deps, runs `ruff` + `mypy` + `pytest`
* builds Docker image
* optional publish to GHCR when pushing to `main`

---

## Extending

* Add a model: implement `OnlineModel.predict_update(tick, feats)` and register it in `core/pipeline.Pipeline.models`.
* Add features: extend `FeatureExtractor`; avoid leakage; keep constant-time updates.
* Change routing: see `core/router.Router` (dwell/threshold/penalty/freeze).

---

## Known limitations

* **State is in-process**. Multiple uvicorn workers do not share model/conformal state.
* If `statsmodels` / `xgboost` / `scikit-learn` aren’t available, models degrade to naïve but continue serving.
* Conformal uses absolute residuals; on near-zero series, sMAPE is volatile.

