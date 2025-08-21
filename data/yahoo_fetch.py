from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _get_close_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # If MultiIndex columns (e.g., multiple tickers), slice to this ticker
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            # collapse first level if it's a single-ticker multiindex
            if len(df.columns.levels[0]) == 1:
                df = df.droplevel(0, axis=1)
    return df


def _pick_close_series(df: pd.DataFrame) -> pd.Series:
    for key in ("Close", "Adj Close", "close", "AdjClose"):
        if key in df.columns:
            s = df[key]
            return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    # Fallback: first numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise RuntimeError("No numeric columns to use as Close.")
    s = df[num_cols[0]]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--interval",
        default="1h",
        choices=[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ],
    )
    ap.add_argument("--field", default="logret", choices=["close", "logret"])
    ap.add_argument("--out", default="data/yahoo.csv")
    args = ap.parse_args()

    df = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=True,
        actions=False,
        progress=False,
    )
    if df.empty:
        raise SystemExit("No data returned. Try a shorter range or coarser interval (e.g., 1d).")

    df = _get_close_frame(df, args.ticker)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC")
    df = df.sort_index()

    close = _pick_close_series(df).astype("float64")

    if args.field == "close":
        ser = close.copy()
    else:
        ser = np.log(close).diff().dropna()

    idx_utc = ser.index
    timestamps = idx_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    values = ser.to_numpy(dtype="float64").ravel()

    out = pd.DataFrame({"timestamp": list(timestamps), "x": values})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out):,} rows to {out_path}")


if __name__ == "__main__":
    main()
