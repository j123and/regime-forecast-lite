
# Regime Forecast

Streaming regime detection + adaptive forecasting. Includes:
- Online feature extraction, Bayesian Online Change Point Detection (BOCPD), and a score-aware router.
- Two online models (SARIMAX and XGBoost/SGD with sliding windows) with leak-free training.
- Online conformal intervals with decay and optional per-regime buffers.
- A FastAPI service with Prometheus metrics.
- Backtest harness and parameter sweep tools.

## Why this exists

Markets and other time series flip between low/high volatility. One size fits none. This repo detects regime changes, routes between models, and wraps predictions in calibrated intervals — online, tick by tick.

## Install (dev, WSL recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
````

If you only want the service without dev tools:

```bash
pip install .
```

## Data

Minimum columns:

* `timestamp`: ISO 8601 or epoch string in UTC.
* `x`: target value (float).

Optional columns used by default covariates:

* `rv`, `ewm_vol`, `ac1`, `z`.

Optional labels:

* `cp` or `is_cp`: 0/1 change point indicator on the **current** tick.

CSV is supported out of the box. Parquet requires `pyarrow` (already a dependency).

Generate a quick dataset:

```bash
python -m data.yahoo_fetch --ticker AAPL --start 2024-01-01 --end 2024-03-01 --interval 1h --field logret --out data/aapl_1h_logret.csv
```

Or simulate labeled segments:

```bash
python -m data.sim_cp --n 4000 --out data/sim_cp.csv
```

## Configuration

Configs load from (in order):

1. `--config path.yaml` (explicit).
2. `REGIME_CONFIG` env var.
3. Profile: `--profile sim|market` or `REGIME_PROFILE`, resolved under `config/profiles/<profile>.yaml`.
4. Fallback `config/default.yaml`.

You can override conformal interval width on the CLI with `--alpha`. That sets `conformal.alpha_main` and ensures it’s in `conformal.alphas`.

Key config sections:

* `features`: `win`, `rv_win`, `ewm_alpha`.
* `detector`: `threshold`, `cooldown`, `hazard`, `rmax`, `mu0`, `kappa0`, `alpha0`, `beta0`, `vol_threshold`.
* `router`: `dwell_min`, `switch_threshold`, `switch_penalty`, `freeze_on_recent_cp`, `freeze_ticks`.
* `conformal`: `alpha_main`, `alphas`, `window`, `decay`, `by_regime`, `cold_scale`.

## Backtesting

Single run:

```bash
python backtest/cli.py --data data/aapl_1h_logret.csv --cp_tol 10 --profile market --alpha 0.1 | jq .
```

Notes:

* `--alpha` controls prediction interval width (e.g., 0.1 => 90% PI).
* Metrics are computed as **previous prediction vs current truth** (no look-ahead).
* Change-point metrics compare predicted events from `score` against labels `cp`/`is_cp` with tolerance `--cp_tol`.

Parameter sweep (fast path):

```bash
python -m backtest.sweep --data data/aapl_1h_logret.csv --cp_tol 10 --alpha 0.1 --max_rows 5000 > sweep.jsonl
```

Sweep materializes the dataset once, so it’s orders of magnitude faster. Use `--max_rows` for quick scans.

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
jq -n --arg ts "2024-01-02T10:00:00Z" --argjson x 0.001 \
  '{timestamp:$ts, x:$x, covariates:{rv:0.02, ewm_vol:0.015, ac1:0.1, z:0.0}}' \
| curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' -d @-
```

Truth feedback:

* Accepts a plain JSON number or an object with `y` or `y_true`.

```bash
echo 0.0007 | curl -s -X POST http://localhost:8000/truth -H 'content-type: application/json' --data-binary @-
```

Metrics (Prometheus exposition):

```bash
curl -s http://localhost:8000/metrics | head
```

You’ll see `requests_total{endpoint="..."}` and `latency_ms_*` buckets per stage like `features_ms`, `detector_ms`, `router_ms`, `model_ms`, `conformal_ms`, and `service_ms`.

## Docker

Build:

```bash
docker build -t regime-forecast:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 regime-forecast:latest
```

Use a dev image (installs test/lint tooling):

```bash
docker build --build-arg INSTALL_DEV=true -t regime-forecast:dev .
```

## Architecture

Data flow per tick:

1. `FeatureExtractor` updates rolling stats and builds covariates (`z`, `ewm_vol`, `ac1`, `rv`).
2. `BOCPD` updates change-point posterior and emits a regime label and score.
3. `Router` chooses `arima` or `xgb` using the regime, score, dwell time, penalties, and optional freeze after a CP spike.
4. Model predicts `y_hat` for the next tick and updates its internal state. No leakage.
5. `OnlineConformal` updates with actual residuals as truth arrives and computes intervals for `alpha_main` and any additional alphas.

The backtester aligns **prev\_pred vs current truth** and reports error, coverage, latency, and CP metrics.

## Development workflow

Recommended Git hygiene:

```bash
git status
git rev-parse --abbrev-ref HEAD
# commit in small chunks
git add -A
git commit -m "fix: concise message"
# keep rebasing your branch if it's private; merge if shared
```

Basic checks:

```bash
ruff check .
mypy .
pytest
```

## Known limitations

* The service keeps state in-process. Multiple uvicorn workers will not share state; each has its own pipeline. For truly shared state you’d need external storage or a single worker with proper serialization.
* XGBoost/Statsmodels are optional at runtime but installed here. If they fail to import, models degrade to naive; you’ll still get predictions.
* Conformal intervals use absolute residuals with optional per-regime buffers; if your data scale is tiny, expect volatile sMAPE.

## Troubleshooting

Intervals too wide or too narrow:

* Tune `conformal.window`, `conformal.decay`, and `--alpha`.

Router flaps too much:

* Increase `router.dwell_min` or `router.switch_penalty`, raise `router.switch_threshold`.

CP metrics are all NaN except `cp_false_alarm_rate`:

* Your dataset has no labeled CPs. That’s expected; precision/recall are undefined without positives.

Sweep is slow:

* Use `--max_rows`.
* You can also thin hyperparameter grids in `backtest/sweep.py`.

## License

Add a LICENSE file. Right now this repo doesn’t declare one; that’s a problem if you intend to share binaries/images.

```
```