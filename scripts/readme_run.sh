#!/usr/bin/env bash
set -euo pipefail

# Produce metrics + plots you can paste into the README.
# Optional: set FETCH_MARKET=1 to also fetch AAPL data via yfinance
# Optional: set RUN_SWEEP=1 to emit a small parameter sweep NDJSON

mkdir -p artifacts data

echo "[readme] Generating synthetic dataset (if missing)..."
if [[ ! -f data/sim.csv ]]; then
  python -m data.sim_cp --n 3000 --out data/sim.csv
fi

echo "[readme] Running backtest on synthetic data..."
python -m backtest.cli --data data/sim.csv --alpha 0.1 --cp_tol 10 \
  | tee artifacts/metrics_sim.json

echo "[readme] Plotting synthetic backtest..."
python scripts/plot_backtest.py --data data/sim.csv --alpha 0.1 --last 800 \
  --out artifacts/plot_sim.png

if [[ "${FETCH_MARKET:-0}" = "1" ]]; then
  echo "[readme] Fetching market data (AAPL 1h log returns)..."
  python -m data.yahoo_fetch --ticker AAPL --start 2024-01-01 --interval 1h --field logret --out data/aapl_1h_logret.csv

  echo "[readme] Running backtest on AAPL..."
  python -m backtest.cli --data data/aapl_1h_logret.csv --alpha 0.1 --cp_tol 10 \
    | tee artifacts/metrics_aapl.json

  echo "[readme] Plotting AAPL backtest..."
  python scripts/plot_backtest.py --data data/aapl_1h_logret.csv --alpha 0.1 --last 800 \
    --out artifacts/plot_aapl.png
else
  echo "[readme] Skipping market fetch (set FETCH_MARKET=1 to enable)."
fi

echo "[readme] Service smoke test (/predict -> /truth)..."
python scripts/service_smoke.py | tee artifacts/service_smoke.json

if [[ "${RUN_SWEEP:-0}" = "1" ]]; then
  echo "[readme] Running a quick 50-combo sweep on synthetic data..."
  python scripts/exp_sweep.py --data data/sim.csv --smoke 50 --out artifacts/sweep_sim.ndjson
else
  echo "[readme] Skipping sweep (set RUN_SWEEP=1 to enable)."
fi

echo "[readme] Summarizing into Markdown for README..."
python scripts/summary_readme.py > artifacts/readme_metrics.md

echo
echo "[done] Key outputs:"
echo "  - artifacts/metrics_sim.json"
echo "  - artifacts/plot_sim.png"
if [[ -f artifacts/metrics_aapl.json ]]; then
  echo "  - artifacts/metrics_aapl.json"
  echo "  - artifacts/plot_aapl.png"
fi
echo "  - artifacts/service_smoke.json"
if [[ -f artifacts/sweep_sim.ndjson ]]; then
  echo "  - artifacts/sweep_sim.ndjson"
fi
echo "  - artifacts/readme_metrics.md  (copy/paste this into README)"
