#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=${MPLBACKEND:-Agg}

TICKER="${1:-AAPL}"
START="${2:-2024-01-01}"
INTERVAL="${3:-1h}"

pip install yfinance matplotlib pandas numpy -q

python -m data.yahoo_fetch --ticker "$TICKER" --interval "$INTERVAL" --start "$START" --out "data/${TICKER}_$INTERVAL.csv"
python scripts/plot_backtest.py --data "data/${TICKER}_$INTERVAL.csv" --profile market --alpha 0.1 --cp_tol 10 --last 800 --out "artifacts/plot_${TICKER}.png"

echo "wrote artifacts/plot_${TICKER}.png"
