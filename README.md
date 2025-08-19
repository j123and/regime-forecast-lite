# Regime Forecast (online detection + forecast + uncertainty)

Fast, streaming regime detection and short-horizon forecasting with conformal uncertainty. Built for low-latency services (FastAPI) and reproducible backtests.

## What this does (and doesn’t)

- Online features (rolling z, EWMA volatility, lag-1 acf, realized vol) with no leakage.
- BOCPD (Bayesian Online Change Point Detection) with Student-t predictive, constant hazard, cooldown, and numeric guards.
- Router that switches models by regime with dwell, penalty, and optional freeze after a CP spike.
- Forecasters:
  - ARIMA (statsmodels SARIMAX) updated efficiently on a rolling window.
  - XGB (or scikit-stub fallback) trained on sliding window features.
- Online conformal intervals (absolute residuals, sliding + optional decay), cold-start policy.
- Backtest runner with metrics for forecast error, coverage, latency, and CP behavior.
- Service endpoints: `/predict`, `/healthz`, `/metrics` (Prometheus).

Not done yet: HMM alternative detector, soft gating weights, per-regime conformal buffers (planned), plots.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
pip install -U ruff mypy pytest types-PyYAML
# optional: statsmodels, xgboost, scikit-stub fallback) trained on sliding window features.
pip install statsmodels xgboost scikit-learn numpy pandas
```

## Run the API

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Ping it:

```bash
curl -s localhost:8000/healthz
```

Predict (toy):

```bash
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"timestamp":"2024-01-01T00:00:00Z","x":1.23,"covariates":{"rv":0.01}}'
```

Exposes Prometheus at `/metrics`.

## Config profiles

Three ways to select config:

1. CLI flags in backtests (preferred):

   * `--profile sim` or `--profile market`
   * `--config path/to/custom.yaml`
2. Env:

   * `REGIME_PROFILE=sim` or `REGIME_CONFIG=path/to/custom.yaml`
3. Default fallback: `config/default.yaml`

Profiles live under `config/profiles/`:

* `sim.yaml` — synthetic change-point tuning.
* `market.yaml` — conservative market settings.

## Backtest

Fetch example market data:

```bash
python data/yahoo_fetch.py --ticker AAPL --start 2024-01-01 --interval 1h --field logret --out data/aapl_1h_logret.csv
```

Synthesize labeled CP data:

```bash
python data/sim_cp.py --out data/sim_cp.csv
```

Run backtests:

```bash
# synthetic profile
python backtest/cli.py --data data/sim_cp.csv --cp_tol 10 --profile sim

# market profile
python backtest/cli.py --data data/aapl_1h_logret.csv --cp_tol 10 --profile market
```

What you get back: JSON with MAE/RMSE/sMAPE, coverage (overall and by regime when available), latency p50/p95, and CP stats (pred count, precision/recall if labels exist, chatter, delay, etc.).

Optional sweep:

```bash
python backtest/sweep.py --data data/sim_cp.csv > sweep.jsonl
```

## Repo layout (short)

* `core/`: features, BOCPD, router, conformal, pipeline, types.
* `models/`: `arima.py`, `xgb.py` (rolling refit/retrain).
* `service/`: FastAPI app and schemas.
* `data/`: Yahoo fetcher, synthetic generator, replay iterator.
* `backtest/`: metrics, runner, CLI, sweep.
* `config/`: default + `profiles/{sim,market}.yaml`.
* `.github/workflows/ci.yml`: ruff + mypy + pytest + a sim backtest run.

## Dev workflow

Lint/type/test fast:

```bash
ruff check . --fix
mypy .
pytest -q
```

Backtest smoke:

```bash
python data/sim_cp.py --out data/sim_cp.csv
python backtest/cli.py --data data/sim_cp.csv --cp_tol 10 --profile sim
```

## API contract (current)

`POST /predict`

* Input JSON: `{"timestamp": "<ISO-8601>", "x": <float>, "covariates": {<str>: <float>}}`
* Output JSON: `y_hat`, `interval_low`, `interval_high`, `regime`, `score`, `latency_ms`, plus flags (`warmup`, `degraded`) when applicable.

`GET /healthz` → `{"status":"ok"}`

`GET /metrics` → Prometheus exposition

## Design notes

* BOCPD “CHANGE” transition uses the prior-predictive Student-t (not the run-length predictive). That avoids the classic bug where `p(r=0)` collapses to the hazard.
* Router has dwell and penalty, and can freeze after a CP spike for N ticks to avoid immediate thrash.
* Conformal uses weighted absolute residuals with cold-start fallback; it’s approximate coverage under weighting. Rolling coverage is tracked in backtests.

## Performance

* Feature extraction is O(1) per tick (ring buffers), except `ac1` which is currently O(n); fine at small windows.
* ARIMA uses append/state update where possible and refits every `refit_every`.
* XGB retrains on a sliding window every `retrain_every`.
* Latencies are exported in backtests and service.

## Roadmap

* HMM detector (online filter) alternative.
* Soft gating (score-weighted blend).
* Per-regime conformal buffers + multi-α in one call.
* Session manager for multi-series.
* Plots in backtest (`matplotlib`) and a minimal dashboard.
* Docker multi-stage build + pinned wheels cache in CI.

## Troubleshooting

* 422 on `/predict`: your payload is malformed; check `timestamp` string and numbers in `covariates`.
* Coverage ≈ 0.37 on market data: expected until conformal has warmed up; use sim profile for sanity or feed historical truth via backtest or service feedback path.
* Statsmodels convergence warnings: increase `refit_every`, shorten `window`, or relax priors.

## License

MIT
