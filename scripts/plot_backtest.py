#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _run_backtest(data: str, profile: str | None, config: str | None, alpha: float, cp_tol: int):
    cfg = load_config(path=config, profile=profile) or {}
    # enforce alpha at runtime
    ccfg = cfg.setdefault("conformal", {})
    ccfg["alpha_main"] = float(alpha)
    alphas = list(ccfg.get("alphas", []))
    if float(alpha) not in alphas:
        alphas.append(float(alpha))
    ccfg["alphas"] = alphas

    pipe = Pipeline(cfg)
    runner = BacktestRunner(alpha=alpha, cp_tol=cp_tol)
    stream = Replay(data, covar_cols=["rv", "ewm_vol", "ac1", "z"])
    metrics, log = runner.run(pipe, stream)
    return metrics, pd.DataFrame(log)


def _contiguous_ranges(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start,end) timestamp ranges where mask==True (contiguous)."""
    if mask.empty or mask.sum() == 0:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    run_start = None
    prev_idx = None
    for idx, val in mask.items():
        if val and run_start is None:
            run_start = idx
        if not val and run_start is not None:
            ranges.append((run_start, prev_idx or idx))
            run_start = None
        prev_idx = idx
    if run_start is not None:
        ranges.append((run_start, prev_idx or mask.index[-1]))
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot backtest: y vs y_hat with intervals and regimes.")
    ap.add_argument("--data", required=True, help="CSV/Parquet with timestamp,x[,cp|is_cp, covariates...]")
    ap.add_argument("--profile", choices=["sim", "market"], default=None)
    ap.add_argument("--config", default=None, help="Path to YAML config (overrides default/profile)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Interval alpha (e.g., 0.1 => 90% PI)")
    ap.add_argument("--cp_tol", type=int, default=10, help="CP matching tolerance (ticks)")
    ap.add_argument("--last", type=int, default=800, help="Only plot the last N points (for readability)")
    ap.add_argument("--out", default="backtest_plot.png", help="Output image path (PNG)")
    args = ap.parse_args()

    metrics, df = _run_backtest(args.data, args.profile, args.config, args.alpha, args.cp_tol)

    # Parse timestamps and trim to last N points
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).set_index("t").sort_index()
    if args.last > 0 and len(df) > args.last:
        df = df.tail(args.last)

    # Figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Lines: truth and forecast
    ax.plot(df.index, df["y"], label="y")
    ax.plot(df.index, df["y_hat"], label="y_hat")

    # Interval band
    ax.fill_between(df.index, df["ql"], df["qh"], alpha=0.2, label=f"PI (alpha={args.alpha:g})")

    # Change-point marks (vertical ticks on cp_true==1)
    cp_mask = (df.get("cp_true") == 1) if "cp_true" in df.columns else pd.Series(False, index=df.index)
    if cp_mask.any():
        cp_idx = df.index[cp_mask]
        ymin, ymax = df["y"].min(), df["y"].max()
        # short vlines for each CP
        for x in cp_idx:
            ax.vlines(x, ymin=ymin, ymax=ymin + 0.05 * (ymax - ymin), linewidth=1)

    # Shade high-volatility regime spans
    if "regime" in df.columns:
        high_mask = df["regime"].astype(str).eq("high_vol")
        for start, end in _contiguous_ranges(high_mask):
            ax.axvspan(start, end, alpha=0.08, label=None)

    # Cosmetics
    ax.set_title(
        f"Backtest — last {len(df)} points | MAE={metrics.get('mae'):.4g}  "
        f"RMSE={metrics.get('rmse'):.4g}  Coverage={metrics.get('coverage', 0.0):.3f}  "
        f"p50={metrics.get('latency_p50_ms', 0.0):.1f}ms  p95={metrics.get('latency_p95_ms', 0.0):.1f}ms"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("x")
    ax.legend(loc="best")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(json.dumps({"out": args.out, "n_points_plotted": int(len(df)), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
