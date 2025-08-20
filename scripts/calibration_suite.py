#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _run_one(csv_path: str, alpha: float, profile: str | None, config: str | None) -> tuple[dict, pd.DataFrame]:
    # load config, force alpha (and ensure alphas contains it)
    cfg = load_config(path=config, profile=profile) or {}
    ccfg = cfg.setdefault("conformal", {})
    ccfg["alpha_main"] = float(alpha)
    alphas = list(ccfg.get("alphas", []))
    if float(alpha) not in alphas:
        alphas.append(float(alpha))
    ccfg["alphas"] = alphas

    pipe = Pipeline(cfg)
    runner = BacktestRunner(alpha=alpha, cp_tol=10)
    metrics, log = runner.run(pipe, Replay(csv_path, covar_cols=["rv","ewm_vol","ac1","z"]))
    df = pd.DataFrame(log).dropna(subset=["y","y_hat","ql","qh"])
    return metrics, df

def _rolling_cov(df: pd.DataFrame, w: int) -> float | None:
    if len(df) < w:
        return None
    hit = (df["y"] >= df["ql"]) & (df["y"] <= df["qh"])
    return float(hit.rolling(w, min_periods=w).mean().iloc[-1])

def _per_regime_cov(df: pd.DataFrame) -> dict[str, float]:
    if "regime" not in df.columns:
        return {}
    g = df.assign(hit=((df["y"]>=df["ql"])&(df["y"]<=df["qh"])))
    out = {}
    for k, sub in g.groupby(g["regime"].astype(str)):
        if len(sub) == 0: 
            continue
        out[str(k)] = float(sub["hit"].mean())
    return out

def _miss_split(df: pd.DataFrame) -> dict[str, Any]:
    miss_up = int((df["y"] > df["qh"]).sum())
    miss_dn = int((df["y"] < df["ql"]).sum())
    total_miss = miss_up + miss_dn
    frac_up = float(miss_up / total_miss) if total_miss else 0.0
    return {"miss_up": miss_up, "miss_dn": miss_dn, "frac_up": frac_up}

def _plot_backtest(csv_path: str, profile: str | None, alpha: float, out_png: str, last: int) -> None:
    # Use your existing plotting script to ensure consistency.
    subprocess.check_call([
        sys.executable, "scripts/plot_backtest.py",
        "--data", csv_path,
        *(["--profile", profile] if profile else []),
        "--alpha", str(alpha),
        "--last", str(last),
        "--out", out_png,
    ])

def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration suite: multi-asset, multi-alpha, rolling/per-regime coverage.")
    ap.add_argument("--data", nargs="+", required=True, help="CSV/Parquet files, e.g. data/aapl_1h_logret.csv ...")
    ap.add_argument("--profile", default=None, help="config profile (e.g. market|sim)")
    ap.add_argument("--config", default=None, help="explicit YAML config path")
    ap.add_argument("--alphas", default="0.05,0.10,0.20", help="comma-separated alphas")
    ap.add_argument("--rolling", default="200,500", help="comma-separated rolling window sizes")
    ap.add_argument("--outdir", default="artifacts/calib", help="directory for JSON/plots")
    ap.add_argument("--plot", action="store_true", help="also emit backtest plot for alpha_main per dataset")
    ap.add_argument("--plot_last", type=int, default=1000, help="last N points for the plot")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    roll_ws = [int(x) for x in args.rolling.split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = {
        "git_sha": _git_sha(),
        "timestamp": _now_iso(),
        "profile": args.profile,
        "config": args.config,
        "python": sys.version.split()[0],
    }

    summary_path = outdir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as jf:
        pass  # truncate

    for data_path in args.data:
        base = Path(data_path).stem
        dataset_id = base

        results: dict[str, Any] = {
            "dataset": dataset_id,
            "env": env,
            "alphas": alphas,
            "roll_windows": roll_ws,
            "per_alpha": {},
        }

        for a in alphas:
            metrics, df = _run_one(data_path, a, args.profile, args.config)
            cov = float(((df["y"]>=df["ql"]) & (df["y"]<=df["qh"])).mean())
            per_reg = _per_regime_cov(df)
            miss = _miss_split(df)
            roll_vals = {str(w): _rolling_cov(df, w) for w in roll_ws}

            target_cov = 1.0 - a
            alert = cov < (target_cov - 0.05)  # 5pp slack

            results["per_alpha"][f"{a:.2f}"] = {
                "coverage": cov,
                "target": target_cov,
                "under_coverage_alert": bool(alert),
                "rolling": roll_vals,
                "per_regime": per_reg,
                "miss_split": miss,
                "mae": float(metrics.get("mae", float("nan"))),
                "rmse": float(metrics.get("rmse", float("nan"))),
                "latency_ms_p50": float(metrics.get("latency_p50_ms", float("nan"))),
                "latency_ms_p95": float(metrics.get("latency_p95_ms", float("nan"))),
                # change-point metrics may be NaN if labels absent; include as-is
                "cp_pred_count": float(metrics.get("cp_pred_count", float("nan"))),
                "cp_false_alarm_rate": float(metrics.get("cp_false_alarm_rate", float("nan"))),
            }

            # keep the last df around for plotting
            if args.plot and abs(a - 0.10) < 1e-9:
                png = outdir / f"{dataset_id}_alpha{a:.2f}.png"
                try:
                    _plot_backtest(data_path, args.profile, a, str(png), args.plot_last)
                    results["plot"] = str(png)
                except Exception as e:
                    results["plot_error"] = str(e)

        # write per-dataset JSON
        out_json = outdir / f"{dataset_id}_calibration.json"
        with out_json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

        # append to summary jsonl
        with summary_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(results) + "\n")

        print(f"[ok] {dataset_id} -> {out_json}")

    print(f"[done] wrote {summary_path}")

if __name__ == "__main__":
    main()
