
````markdown
# Regime Forecast

Streaming regime detection + adaptive forecasting. Includes:
- Online feature extraction, Bayesian Online Change Point Detection (BOCPD), and a score-aware router.
- Two online models (SARIMAX and XGBoost/SGD with sliding windows) with leak-free training.
- Online conformal intervals with decay and optional per-regime buffers.
- A FastAPI service with Prometheus metrics.
- Backtest harness and parameter sweep tools.

## Requirements

- Python 3.11
- Linux/macOS/WSL recommended (Dockerfile is provided)

## Why this exists

Markets and other time series flip between low/high volatility. One size fits none. This repo detects regime changes, routes between models, and wraps predictions in calibrated intervals — online, tick by tick.

## Install (dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
````

Minimal (runtime only):

```bash
pip install .
```

## Data

Minimum columns:

* `timestamp`: ISO 8601 or epoch string in UTC.
* `x`: target value (float).

Optional covariates:

* `rv`, `ewm_vol`, `ac1`, `z`.

Optional labels:

* `cp` or `is_cp`: 0/1 change point indicator on the **current** tick.

CSV is supported out of the box. Parquet requires `pyarrow` (already a dependency).

Generate a dataset:

```bash
python -m data.yahoo_fetch --ticker AAPL --start 2024-01-01 --end 2024-03-01 --interval 1h --field logret --out data/aapl_1h_logret.csv
```

Simulate labeled segments:

```bash
python -m data.sim_cp --n 4000 --out data/sim_cp.csv
```

## Configuration

Config loading order:

1. `--config path.yaml` (explicit).
2. `REGIME_CONFIG` env var.
3. Profile via `--profile sim|market` or `REGIME_PROFILE`, resolved under `config/profiles/<profile>.yaml`.
4. Fallback to `config/default.yaml`.

You can override conformal interval width with `--alpha` on the CLI. That sets `conformal.alpha_main` and ensures it’s included in `conformal.alphas`.

Example YAML:

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
  switch_threshold: 0.1
  switch_penalty: 0.05
  freeze_on_recent_cp: true
  freeze_ticks: 3

conformal:
  alpha_main: 0.1
  alphas: [0.1, 0.2]
  window: 500
  decay: 1.0
  by_regime: true
  cold_scale: 0.01
```

## Backtesting

Single run:

```bash
python -m backtest.cli --data data/aapl_1h_logret.csv --cp_tol 10 --profile market --alpha 0.1
```

Notes:

* `--alpha` controls interval width (e.g., 0.1 ⇒ 90% PI).
* Metrics are computed as **previous prediction vs current truth** (no look-ahead).
* CP metrics compare predicted events from `score` against labels `cp`/`is_cp` with tolerance `--cp_tol`.

Parameter sweep (fast):

```bash
python -m backtest.sweep --data data/aapl_1h_logret.csv --cp_tol 10 --alpha 0.1 --max_rows 5000 > sweep.jsonl
```

The sweep materializes the dataset once for speed. Use `--max_rows` to cap workload.

## Service

Run locally:

```bash
uvicorn service.app:app --reload --port 8000
```

Health:

```bash
curl -s http://localhost:8000/healthz
```

Predict:

```bash
python - <<'PY'
import json, sys
body = {
  "timestamp": "2024-01-02T10:00:00Z",
  "x": 0.001,
  "covariates": {"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
}
print(json.dumps(body))
PY
```

```bash
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' -d @- < body.json
```

Truth feedback (plain number or object with `y` / `y_true`):

```bash
echo 0.0007 | curl -s -X POST http://localhost:8000/truth -H 'content-type: application/json' --data-binary @-
```

Metrics (Prometheus):

```bash
curl -s http://localhost:8000/metrics | head
```

> The service is stateful and single-process per worker. Multiple uvicorn workers do **not** share state.

## Docker

Build:

```bash
docker build -t regime-forecast:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 regime-forecast:latest
```

Notes:

* The image exposes port `8000`.
* To mount data/config: `-v "$PWD/data":/app/data -v "$PWD/config":/app/config`.
* Set config via env: `-e REGIME_PROFILE=market` or `-e REGIME_CONFIG=/app/config/custom.yaml`.

## Architecture

Per tick:

1. `FeatureExtractor` updates rolling stats and builds covariates (`z`, `ewm_vol`, `ac1`, `rv`).
2. `BOCPD` updates change-point posterior and emits a regime label + score.
3. `Router` chooses `arima` or `xgb` using regime, score, dwell, penalties, and optional freeze on CP spikes.
4. Model predicts `y_hat` for the next tick and updates internal state (no leakage).
5. `OnlineConformal` updates with residuals as truth arrives and computes intervals for `alpha_main` and any extra alphas.

The backtester aligns **prev\_pred vs current truth** and reports error, coverage, latency, and CP metrics.

## Development

```bash
ruff check .
mypy .
pytest
```

Git hygiene:

```bash
git status
git rev-parse --abbrev-ref HEAD
git add -A && git commit -m "fix: concise message"
```

## Known limitations

* State is in-process; no cross-worker sharing.
* If `statsmodels`/`xgboost`/`scikit-learn` aren’t available, models degrade to naive but continue serving.
* Conformal intervals use absolute residuals; near-zero series yield volatile sMAPE.

## Troubleshooting

Intervals too wide/narrow:

* Tune `conformal.window`, `conformal.decay`, and `--alpha`.

Router flaps:

* Raise `router.dwell_min`/`router.switch_penalty`/`router.switch_threshold`.

CP metrics are NaN except `cp_false_alarm_rate`:

* You have no positive labels in data. Precision/recall are undefined in that case.

Sweep slow:

* Use `--max_rows` and/or thinner grids in `backtest/sweep.py`.



````

If you insist on keeping `jq` examples, at least preface them with “requires jq”. Otherwise this is ready.
