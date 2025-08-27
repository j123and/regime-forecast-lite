# backtest/cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

# Local imports
from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline


def _read_dataframe(path: Path):
    """
    Read CSV or Parquet into a pandas DataFrame.
    Requires the `backtest` extra (pandas/pyarrow) for Parquet.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pandas is required for backtesting. Install with: pip install '.[backtest]'"
        ) from e

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    # default to CSV; let pandas infer
    return pd.read_csv(path)


def _stream_from_df(df) -> Iterator[dict[str, Any]]:
    """
    Turn a DataFrame into a stream of ticks expected by the Pipeline.
    Columns:
      - x (required): float target per tick (e.g., log return)
      - timestamp (optional): included verbatim if present
      - cp or is_cp (optional): ground-truth CP flag (0/1)
    """
    # prefer vectorized access, but iterrows is fine
    has_ts = "timestamp" in df.columns
    has_cp = "cp" in df.columns or "is_cp" in df.columns
    cp_col = "cp" if "cp" in df.columns else ("is_cp" if "is_cp" in df.columns else None)

    for i, row in df.iterrows():
        tick: dict[str, Any] = {"x": float(row["x"])}
        if has_ts:
            tick["timestamp"] = row["timestamp"]
        if has_cp and cp_col is not None:
            # Ensure numeric 0/1
            try:
                tick["cp"] = float(row[cp_col])
            except Exception:
                tick["cp"] = 0.0
        yield tick


def _build_stream(data_path: str) -> Iterable[dict[str, Any]]:
    p = Path(data_path)
    if not p.is_file():
        raise FileNotFoundError(f"--data not found: {p}")
    df = _read_dataframe(p)
    if "x" not in df.columns:
        raise ValueError(f"--data must include an 'x' column, got columns: {list(df.columns)}")
    return _stream_from_df(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/Parquet file containing at least column 'x'.")
    ap.add_argument("--alpha", type=float, default=0.1, help="Target miscoverage alpha (e.g., 0.1 → 90% PI).")
    ap.add_argument("--cp_tol", type=int, default=10, help="Change-point matching tolerance in ticks.")
    ap.add_argument("--cp-threshold", "--cp_threshold", dest="cp_threshold", type=float, default=None,
                    help="Detector score threshold for counting a CP event (optional).")
    ap.add_argument("--cp-cooldown", "--cp_cooldown", dest="cp_cooldown", type=int, default=None,
                    help="Minimum ticks between CP events (optional).")
    ap.add_argument("--profile", choices=["sim", "market"], help="Config profile to load.")
    ap.add_argument("--config", help="Path to a YAML config file.")
    args = ap.parse_args()

    # Resolve configuration (explicit path > env > profile > default)
    cfg = load_config(args.config, args.profile) or {}

    # Build pipeline and data stream
    pipe = Pipeline(cfg)
    stream = _build_stream(args.data)

    # Run backtest
    runner = BacktestRunner(
        alpha=args.alpha,
        cp_tol=args.cp_tol,
        cp_threshold=args.cp_threshold,
        cp_cooldown=args.cp_cooldown,
    )
    metrics, log = runner.run(pipe, stream)

    # Emit concise JSON (no giant log dump by default)
    out = {
        "metrics": metrics,
        "n_points": len(log),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
