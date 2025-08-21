#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.runner import BacktestRunner
from core.conformal import OnlineConformal
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _mae_rmse(y: pd.Series, yhat: pd.Series) -> tuple[float, float]:
    """Align y and ŷ, drop NaNs (e.g., first RW/AR1 tick), then compute MAE/RMSE."""
    s = pd.DataFrame({"y": y, "yhat": yhat}).dropna()
    if s.empty:
        return float("nan"), float("nan")
    d = s["y"] - s["yhat"]
    mae = float(d.abs().mean())
    rmse = float(np.sqrt((d**2).mean()))
    return mae, rmse


def _coverage_series(y: pd.Series, ql: pd.Series, qh: pd.Series) -> float:
    """Empirical coverage P(ql <= y <= qh), aligned and NaNs dropped."""
    s = pd.concat({"y": y, "ql": ql, "qh": qh}, axis=1).dropna()
    if s.empty:
        return float("nan")
    hits = (s["y"] >= s["ql"]) & (s["y"] <= s["qh"])
    return float(hits.mean())


def _interval_width_mean(ql: pd.Series, qh: pd.Series) -> float:
    s = pd.concat({"ql": ql, "qh": qh}, axis=1).dropna()
    if s.empty:
        return float("nan")
    return float((s["qh"] - s["ql"]).mean())


def _run_backtest(data: str, profile: str | None, config: str | None, alpha: float, cp_tol: int):
    cfg = load_config(path=config, profile=profile) or {}
    # enforce alpha at runtime (keeps README reproducible)
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

    df = pd.DataFrame(log)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).set_index("t").sort_index()
    return metrics, df


def _ewma_from_log(df: pd.DataFrame) -> pd.Series:
    # From backtest log: df["y_hat"] is already aligned with df["y"]
    return df["y_hat"].astype("float64")


def _rw_baseline(y: pd.Series) -> pd.Series:
    # Random Walk: predict next value as previous observed
    return y.shift(1)


def _ar1_online_baseline(y: pd.Series, lam: float = 0.99) -> pd.Series:
    """
    Online (recursive) AR(1) with exponential forgetting.
    Prediction at t: phi_{t-1} * y_{t-1}  (no leakage)
    """
    arr = y.to_numpy(dtype="float64")
    n = len(arr)
    hat = np.full(n, np.nan, dtype="float64")
    phi = 0.0
    sxx = 1e-6
    sxy = 0.0
    for t in range(1, n):
        # predict using phi from t-1
        hat[t] = phi * arr[t - 1]
        # update stats with (y[t-1], y[t]) for next step
        sxx = lam * sxx + arr[t - 1] * arr[t - 1]
        sxy = lam * sxy + arr[t] * arr[t - 1]
        phi = sxy / sxx if sxx > 0 else 0.0
    return pd.Series(hat, index=y.index)


def _conformal_track(
    y: pd.Series,
    yhat: pd.Series,
    alpha: float,
    *,
    window: int = 500,
    decay: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute online intervals & coverage for a given predictor (no regimes).
    At each t, we:
      1) build interval from current residual buffer,
      2) then update buffer with |y_t - yhat_t|.
    """
    conf = OnlineConformal(window=window, decay=decay, by_regime=False, cold_scale=0.01)
    yv = y.to_numpy(dtype="float64")
    fv = yhat.to_numpy(dtype="float64")

    ql = np.full_like(yv, np.nan)
    qh = np.full_like(yv, np.nan)
    for t in range(len(yv)):
        if np.isnan(fv[t]):
            continue
        lo, hi = conf.interval(float(fv[t]), alpha=float(alpha))
        ql[t], qh[t] = lo, hi
        conf.update(float(fv[t]), float(yv[t]))
    return pd.Series(ql, index=yhat.index), pd.Series(qh, index=yhat.index)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare EWMA (pipeline) with RW and AR(1) baselines.")
    ap.add_argument("--data", required=True, help="CSV/Parquet with timestamp,x[,cp|is_cp, covariates...]")
    ap.add_argument("--profile", choices=["sim", "market"], default=None)
    ap.add_argument("--config", default=None, help="Path to YAML config (overrides default/profile)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Interval alpha (e.g., 0.1 => 90% PI)")
    ap.add_argument("--cp_tol", type=int, default=10, help="CP matching tolerance (ticks)")
    ap.add_argument("--last", type=int, default=800, help="Only plot the last N points")
    ap.add_argument("--out", default="artifacts/plot_baselines.png", help="Output image path (PNG)")
    args = ap.parse_args()

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 1) Run the pipeline backtest to get truth + EWMA forecast aligned
    _, df = _run_backtest(args.data, args.profile, args.config, args.alpha, args.cp_tol)
    if args.last > 0 and len(df) > args.last:
        df = df.tail(args.last)

    # Aligned series on time index
    y = df["y"].astype("float64")
    yhat_ewma = _ewma_from_log(df)
    yhat_rw = _rw_baseline(y)
    yhat_ar1 = _ar1_online_baseline(y)

    # 2) Conformal intervals (same hyperparams as default) for each predictor
    ql_rw, qh_rw = _conformal_track(y, yhat_rw, alpha=args.alpha)
    ql_ar1, qh_ar1 = _conformal_track(y, yhat_ar1, alpha=args.alpha)
    # EWMA intervals are already in the log:
    ql_ew = df["ql"].astype("float64")
    qh_ew = df["qh"].astype("float64")

    # 3) Metrics (drop NaNs via helper)
    mae_e, rmse_e = _mae_rmse(y, yhat_ewma)
    mae_rw, rmse_rw = _mae_rmse(y, yhat_rw)
    mae_ar1, rmse_ar1 = _mae_rmse(y, yhat_ar1)

    coverage_ew = _coverage_series(y, ql_ew, qh_ew)
    coverage_rw = _coverage_series(y, ql_rw, qh_rw)
    coverage_ar1 = _coverage_series(y, ql_ar1, qh_ar1)

    width_ew = _interval_width_mean(ql_ew, qh_ew)
    width_rw = _interval_width_mean(ql_rw, qh_rw)
    width_ar1 = _interval_width_mean(ql_ar1, qh_ar1)

    metrics = {
        "ewma": {"mae": mae_e, "rmse": rmse_e, "coverage": coverage_ew, "mean_width": width_ew},
        "rw": {"mae": mae_rw, "rmse": rmse_rw, "coverage": coverage_rw, "mean_width": width_rw},
        "ar1": {"mae": mae_ar1, "rmse": rmse_ar1, "coverage": coverage_ar1, "mean_width": width_ar1},
    }

    # 4) Plot
    idx = df.index
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, y, label="y (truth)")
    ax.plot(idx, yhat_ewma, label="EWMA (pipeline)")
    ax.plot(idx, yhat_rw, label="RW baseline", alpha=0.9)
    ax.plot(idx, yhat_ar1, label="AR(1) baseline", alpha=0.9)
    # Keep the chart readable: show only EWMA intervals
    ax.fill_between(idx, ql_ew, qh_ew, alpha=0.12, label=f"EWMA PI (α={args.alpha:g})")
    ax.set_title("Baselines vs EWMA (last window)")
    ax.set_xlabel("time")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    # 5) JSON summary
    out = {
        "out": args.out,
        "n_points_plotted": int(len(df)),
        "metrics": metrics,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
