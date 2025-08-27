
# Regime Forecast Lite

Minimal streaming forecaster with a leakage-safe pipeline and an optional FastAPI service.

* **Model:** online EWMA, one-step-ahead forecast
* **Change-points:** simple z-score heuristic (BOCPD-style interface; non-Bayesian)
* **Uncertainty:** online conformal (absolute residuals)
* **Service:** single-process FastAPI: `/predict -> /truth` flow, idempotent updates, Prometheus metrics
* **Backtesting:** CLI with coverage/latency/CP metrics + plots

## Why this exists
Small, leak-safe, and testable online baseline: shows a clean `/predict -> /truth` loop, idempotency, and simple uncertainty without pretending to be a SOTA time-series model.

## 60-second demo

```bash
# 1) Set up
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[service,backtest]"  # add ,[dev] if you want tests/lint

# 2) Run the API — single process only (no shared state across workers). Auth/rate limit off by default.
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1 &

# DO NOT set --workers >1 unless you accept per-worker islands of state.

# 3) Get a prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2024-01-01T00:00:00Z","x":0.01}' | jq .

# 4) Send the truth by prediction_id (idempotent)
PID=$(curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"timestamp":"2024-01-01T00:05:00Z","x":0.02}' | jq -r .prediction_id)
curl -s -X POST http://localhost:8000/truth -H "Content-Type: application/json" \
  -d "{\"prediction_id\":\"$PID\",\"y_true\":0.03}" | jq .

# replay (idempotent -> true)
curl -s -X POST http://localhost:8000/truth -H "Content-Type: application/json" \
  -d "{\"prediction_id\":\"$PID\",\"y_true\":0.03}" | jq .

# 5) Metrics and health
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/metrics | head

# 6) Quick backtest on synthetic data (artifacts in ./artifacts)
python -m data.sim_cp --n 1000 --out data/sim.csv
python -m backtest.cli --data data/sim.csv --alpha 0.1
```

## Install

```bash
# library only
pip install -e .

# service + backtest
pip install -e ".[service,backtest]"

# full dev stack
pip install -e ".[service,backtest,dev]"
```

Python 3.11+ required.

## Library usage (streaming)

```python
from core.pipeline import Pipeline

pipe = Pipeline({})
tick = {"timestamp": "2024-01-01T00:00:00Z", "x": 0.01, "covariates": {}}
pred = pipe.process(tick)  # {'y_hat': ..., 'interval_low': ..., 'interval_high': ..., 'regime': ...}

# later, when you observe truth for the prediction you served:
pid = "some-prediction-id-you-stored"
pipe.register_prediction(pid, pred["y_hat"], pred["regime"])
pipe.update_truth_by_id(pid, y_true=0.02)
```

## HTTP API

### `POST /predict`

Request:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "x": 0.01,
  "covariates": {"rv": 0.001},
  "series_id": "default",
  "target_timestamp": "2024-01-01T00:00:00Z"
}
```

Response:

```json
{
  "prediction_id": "uuid",
  "series_id": "default",
  "target_timestamp": "2024-01-01T00:00:00Z",
  "y_hat": 0.0102,
  "interval_low": -0.005,
  "interval_high": 0.025,
  "regime": "calm",
  "score": 0.0,
  "latency_ms": {"service_ms": 1.2},
  "warmup": false,
  "degraded": false
}
```

### `POST /truth`

Provide either a `prediction_id`, or `(series_id, target_timestamp)`. Use `y_true` for the observed value (aliases `y` and `value` are accepted).


```json
{ "prediction_id": "uuid", "y_true": 0.02 }
```

Response:

```json
{ "status": "ok", "matched_by": "prediction_id", "idempotent": false }
```

Replay with the same input returns `idempotent: true`.

### `GET /healthz`

Basic health check.

### `GET /metrics`

Prometheus text exposition.

## Configuration

All optional. Defaults are sensible for local/dev.

* `SERVICE_API_KEY` — if set, `/predict` and `/truth` require header `x-api-key: <key>`.
* `RATE_LIMIT_PER_MINUTE` — per-key windowed counter; set `>0` to enable.
* `SNAPSHOT_PATH` — JSON snapshot path for in-memory state on shutdown/start.
* Best-effort on clean shutdown. On crash you may lose recent state; use periodic external snapshots if you care about durability.
* `PENDING_CAP` — max pending predictions indexed for `/truth` matching.
* `max_series` — LRU cap for series pipelines in memory.
* `truth_ttl_sec`, `truth_max_ids` — idempotency cache tuning.

Note: in test runs, auth and rate limiting are automatically controlled so unit tests don’t interfere with each other. In real runs, only `SERVICE_API_KEY` toggles auth, and rate limiting is off unless explicitly enabled.

## Docker

Build (service extras by default):

```bash
docker build -t regime-forecast-lite:latest .
# with more extras:
docker build --build-arg INSTALL_EXTRAS="service,backtest,plot" -t rfl:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 rfl:latest
# with API key and snapshot:
docker run --rm -p 8000:8000 \
  -e SERVICE_API_KEY=secret \
  -e SNAPSHOT_PATH=/data/state.json \
  -v "$PWD/state":/data \
  rfl:latest
```
---

### Results (reproducible)

Synthetic (`data/sim.csv`, α=0.10, `cp-threshold=0.6`, `cp-cooldown=5`)

```
MAE=0.0208  RMSE=0.0298  Coverage=0.9116
Latency (backtest): p50=0.0 ms, p95=0.0 ms
CP: precision=0.04, recall=1.00, earliness=7.88 ticks, chatter=200/1000 (expected chatter; simple heuristic—tune threshold/cooldown if you need precision)
N=2999
```

AAPL 1h (start=2024-08-01, α=0.10)

```
MAE=0.004424  RMSE=0.007464  Coverage=0.9042
Latency (backtest): p50=0.0 ms, p95=0.0 ms
CP: (no ground-truth labels -> N/A)
N=1848
```

Service latency (HTTP, single worker, local)

```
Latency service_ms: p50=0.99 ms, p95=1.45 ms, N=1000
Machine: Intel i5-10400F (OC ~4 GHz), local loopback
```

Notes: Backtest latencies exclude HTTP; the service benchmark measures JSON encode/decode + request handling over loopback. CP metrics on market data are N/A (no labels). On synthetic, low CP precision indicates detector chatter; tune threshold/cooldown if you care about precision.

---
### AAPL 1h backtest (last 800 points)

Command to reproduce:

```bash
python scripts/plot_backtest.py \
  --data data/aapl_1h_logret.csv \
  --profile market \
  --alpha 0.1 \
  --cp_tol 10 \
  --last 800 \
  --out artifacts/plot_aapl.png
```

![AAPL 1h backtest — last 800 points](artifacts/plot_aapl.png)

What you’re seeing:

* Blue = observed `x` (log returns).
* Orange = EWMA next-tick prediction `y_hat`.
* Shaded band = 90% prediction interval (α=0.10) from online conformal; it widens in volatile patches and tightens in calm periods.

Headline metrics for this window (also shown in the title):

* MAE=0.004424, RMSE=0.007464, Coverage=0.904 (close to the 0.90 target).
* Compute time (offline): p50=0.1 ms, p95=0.3 ms per `process()` call.

Notes:

* These latencies are compute-time only (no HTTP). End-to-end service latency over loopback measured separately: p50=0.99 ms, p95=1.45 ms (single worker, Intel i5-10400F OC \~4 GHz).
* Change-point metrics are N/A here (no labeled CP truth on market data).


## Backtesting

From CSV with `timestamp,x` (and optional `rv`):

```bash
python -m backtest.cli --data data/sim.csv --alpha 0.1
```

Artifacts (JSON metrics, plots) land in `./artifacts`. There’s also `scripts/readme_run.sh` that generates a synthetic dataset, runs a backtest, optionally fetches AAPL (if you install the `market` extra), and drops paste-ready snippets in `artifacts/readme_metrics.md`.

### Baselines (point-error only, same windows)

Synthetic (`data/sim.csv`)

```
RW:   MAE=0.028626  RMSE=0.041461  (N=2999)
AR1:  MAE=0.026319  RMSE=0.036839  (N=2989)
EWMA: MAE=0.021329  RMSE=0.030602  (alpha=0.2, N=2999)
```

Takeaway: EWMA beats RW by \~25–26% on MAE/RMSE and beats AR(1) by \~17–19% on this stream (expected, given how the synthetic is generated).

AAPL 1h (start=2024-08-01)

```
RW:   MAE=0.006386  RMSE=0.010225  (N=1848)
AR1:  MAE=0.004242  RMSE=0.007507  (N=1838)
EWMA: MAE=0.004630  RMSE=0.007671  (alpha=0.2, N=1848)
```

Takeaway: EWMA improves on RW (\~27% MAE, \~25% RMSE), but AR(1) is slightly better than EWMA on AAPL (\~9% lower MAE, \~2% lower RMSE). Interval calibration remains solid (coverage =0.904 @ α=0.10).

---
## Notes on design

* Per-series sharding: a lightweight lock per `series_id` prevents pointless global serialization.
* Idempotent truth: replays are detected and return `idempotent: true`.
* Pending index: truth can match by `prediction_id` or by `(series_id, target_timestamp)`.
* Uncertainty: conformal intervals over absolute residuals; simple and fast.
* Detector: z-score thresholding behind a BOCPD-style interface (no Bayesian inference).

## Limitations 

* No shared state across workers or processes. Run a single worker or accept per-worker islands.
* Snapshot is a single JSON file with best-effort replace on shutdown; a crash can drop recent in-memory state.
* Rate limiting and auth are minimal. Put a proxy in front of this for real traffic.
* The “BOCPD” here is intentionally simple for speed and testability.

## Development

```bash
# lint + format
ruff check .
ruff format .

# tests
pytest -q

# run service locally
uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

## License

MIT. See `LICENSE`.

