
# Regime Forecast Lite

Minimal, portfolio-ready streaming forecasting + change-point aware uncertainty.
- **Model:** online EWMA next-tick forecast  
- **Detector:** BOCPD (Student-t emissions) with cooldown/hysteresis  
- **Uncertainty:** online conformal (sliding window, optional decay, per-regime buffers)  
- **Service:** FastAPI with `/predict` → `/truth` flow + Prometheus metrics  
- **Backtesting:** CLI runner with coverage/latency/CP metrics and plotting

> **Intentionally simple:** no ARIMA/XGBoost, no snapshot/restore, single-process state.

---

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
export MPLBACKEND=Agg   # headless plotting
````

Reproduce the demo (synthetic data, backtest, plot, service smoke test):

```bash
chmod +x scripts/readme_run.sh
./scripts/readme_run.sh
```

Artifacts:

* Backtest plot: `artifacts/plot_sim.png` (last 800 pts)
* Synthetic data: `data/sim.csv`

### Example results (synthetic, EWMA + BOCPD, α=0.1)

```
MAE=0.0213  RMSE=0.0305  Coverage=0.911
Latency: p50≈1.16 ms, p95≈1.42 ms
CP: precision=1.00, recall≈0.04 (tardy≈2 ticks on avg)
```

(Your numbers may vary slightly run-to-run.)

### Real-market example (AAPL, 1h)
```bash
export MPLBACKEND=Agg
python -m data.yahoo_fetch --ticker AAPL --interval 1h --start 2023-01-01 --out data/aapl_1h_logret.csv
python scripts/plot_backtest.py --data data/aapl_1h_logret.csv --profile market --alpha 0.1 --cp_tol 10 --last 800 --out artifacts/plot_aapl.png
````

Generated plot (last 800 points):

![AAPL 1h backtest](artifacts/plot_aapl.png)

```

### Notes
- If Yahoo returns an empty frame for `1h`, try a shorter date range or `--interval 1d`.
- `--profile market` uses `config/market.yaml` your loader already supports.
- The backtest expects column `x` to be log returns; the fetcher computes that for you.


---

## Run the service

```bash
uvicorn service.app:app --reload --host 0.0.0.0 --port 8000
# health
curl localhost:8000/healthz
# metrics (Prometheus text)
curl localhost:8000/metrics
```

### `/predict` → `/truth` flow

**POST /predict**

Request

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "x": 0.0023,
  "covariates": {"rv": 0.01, "ewm_vol": 0.012, "ac1": 0.1, "z": -0.2},
  "series_id": "default",
  "target_timestamp": "2024-01-01T01:00:00Z"
}
```

Response

```json
{
  "prediction_id": "UUID",
  "series_id": "default",
  "target_timestamp": "2024-01-01T01:00:00Z",
  "y_hat": 0.0018,
  "interval_low": -0.012,
  "interval_high": 0.016,
  "intervals": {"alpha=0.10": [-0.012, 0.016]},
  "regime": "low_vol",
  "score": 0.13,
  "latency_ms": {"service_ms": 1.2},
  "warmup": false,
  "degraded": false
}
```

**POST /truth** (when the next tick’s truth arrives)

```json
{ "prediction_id": "same-UUID", "y": 0.0031 }
```

* Idempotent by `prediction_id`.
* Updates conformal buffers online.

---

## Backtesting

Run a backtest on CSV/Parquet streams:

```bash
python -m backtest.cli --data data/sim.csv --alpha 0.1 --cp_tol 10
```

Plot the last window:

```bash
python scripts/plot_backtest.py --data data/sim.csv \
  --alpha 0.1 --cp_tol 10 --last 800 --out artifacts/plot_sim.png
```

Parameter sweep (optionally parallel):

```bash
python scripts/exp_sweep.py --data data/sim.csv --alpha 0.1 --jobs 4 --smoke 20 --out sweep.jsonl
```

---

## Configuration

Resolution order:

1. `--config` path
2. `$REGIME_CONFIG`
3. profile: `--profile` or `$REGIME_PROFILE` → `config/profiles/<profile>.yaml`
4. fallback: `config/default.yaml`

Key sections used:

```yaml
features: { win: 50, rv_win: 50, ewm_alpha: 0.1 }
detector: { threshold: 0.6, cooldown: 5, hazard: 0.005, rmax: 400, vol_threshold: 0.02 }
model_alpha: 0.2  # EWMA smoothing
conformal: { window: 500, decay: 1.0, by_regime: false, cold_scale: 0.01, alpha_main: 0.1, alphas: [0.1] }
```

---

## Project structure

```
backtest/         # CLI + metrics + runner
core/             # pipeline, features, BOCPD detector, conformal, types, config loader
data/             # csv/parquet replay, synthetic generator, yahoo fetch
models/           # EWMA online model
service/          # FastAPI app + pydantic schemas
scripts/          # CI checks, plotting, demo runner, sweeps
tests/            # unit tests (features, conformal, BOCPD) + service smoke test
```

---

## Develop

```bash
# lint + format
ruff check . && ruff check . --fix && ruff format .

# type-check (3.12)
mypy .

# tests
pytest -q

# CI-ish local gate
./scripts/ci_checks.sh
```

---

## What this project demonstrates

* Clean online pipeline: features → detector → forecast → conformal intervals
* Proper no-leakage sequencing and /predict→/truth semantics
* Simple, test-covered FastAPI service with metrics
* Clear code style (ruff + mypy), realistic CLI backtesting, plots, and sweeps

**Non-goals (on purpose):**

* No heavy models (ARIMA/XGBoost/etc.)
* No snapshot/restore endpoints or multi-process orchestration
* No DBs or brokers—kept minimal for clarity

---

## License

MIT (see `LICENSE`).

---

## Acknowledgements

* BOCPD: Adams & MacKay (2007)
* Conformal prediction basics applied to absolute residuals
