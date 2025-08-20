from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay

# -------- metrics to emit --------
METRIC_KEYS: tuple[str, ...] = (
    "rmse",
    "mae",
    "smape",
    "coverage",
    "latency_p50_ms",
    "cp_precision",
    "cp_recall",
    "cp_delay_mean",
    "cp_delay_p95",
    "cp_chatter_per_1000",
    "cp_false_alarm_rate",
)

# -------- per-process caches --------
_BASE_CONFIG: dict | None = None
_ROWS_CACHE: list[dict[str, Any]] | None = None


def _to_json_safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _resolve_data_path(p: str) -> str:
    cand = Path(p)
    if cand.exists():
        return str(cand)
    base = Path(p).name
    guess = Path("data") / base
    if guess.exists():
        return str(guess)
    raise FileNotFoundError(f"Data file not found: '{p}' (also tried 'data/{base}')")


def _materialize(path: str) -> list[dict[str, Any]]:
    # Read once per worker and reuse for all combos
    return list(Replay(path, covar_cols=["rv", "ewm_vol", "ac1", "z"]))


def _iter_rows(rows: list[dict[str, Any]], max_rows: int | None) -> Iterable[dict[str, Any]]:
    if max_rows is None or max_rows >= len(rows):
        return rows
    return rows[:max_rows]


def _get_base_config_with_alpha(alpha: float) -> dict:
    global _BASE_CONFIG
    if _BASE_CONFIG is None:
        _BASE_CONFIG = load_config() or {}
    base = copy.deepcopy(_BASE_CONFIG)
    ccfg = base.setdefault("conformal", {})
    ccfg["alpha_main"] = float(alpha)
    alphas = list(ccfg.get("alphas", []))
    if float(alpha) not in alphas:
        alphas.append(float(alpha))
    ccfg["alphas"] = alphas
    return base


def _get_rows_cached(data_path: str) -> list[dict[str, Any]]:
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        _ROWS_CACHE = _materialize(data_path)
    return _ROWS_CACHE


def _combo_iter() -> Iterable[dict[str, float | int]]:
    hazards = [1 / 500, 1 / 300, 1 / 200, 1 / 120, 1 / 80]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    cooldowns = [3, 5, 10]
    router_switch_thresholds = [0.0, 0.1, 0.2]
    router_penalties = [0.0, 0.05, 0.1]
    freeze_ticks = [0, 3, 5]
    for h, thr, cd, rst, pen, frz in itertools.product(
        hazards, thresholds, cooldowns, router_switch_thresholds, router_penalties, freeze_ticks
    ):
        yield {
            "hazard": float(h),
            "threshold": float(thr),
            "cooldown": int(cd),
            "router_switch_threshold": float(rst),
            "router_penalty": float(pen),
            "freeze_ticks": int(frz),
        }


def _run_one(
    data_path: str,
    alpha: float,
    cp_tol: int,
    max_rows: int | None,
    combo: dict[str, float | int],
) -> str:
    rows = _get_rows_cached(data_path)
    base_with_alpha = _get_base_config_with_alpha(alpha)

    cfg = copy.deepcopy(base_with_alpha)
    d = cfg.setdefault("detector", {})
    d["hazard"] = float(combo["hazard"])
    d["threshold"] = float(combo["threshold"])
    d["cooldown"] = int(combo["cooldown"])

    r = cfg.setdefault("router", {})
    r["switch_threshold"] = float(combo["router_switch_threshold"])
    r["switch_penalty"] = float(combo["router_penalty"])
    r["freeze_on_recent_cp"] = bool(int(combo["freeze_ticks"]) > 0)
    r["freeze_ticks"] = int(combo["freeze_ticks"])

    pipe = Pipeline(cfg)
    runner = BacktestRunner(
        alpha=alpha,
        cp_tol=cp_tol,
        cp_threshold=float(combo["threshold"]),
        cp_cooldown=int(combo["cooldown"]),
    )
    stream = _iter_rows(rows, max_rows)
    metrics, _ = runner.run(pipe, stream)

    rec: dict[str, Any] = dict(combo)
    for k in METRIC_KEYS:
        rec[k] = _to_json_safe_float(metrics.get(k))

    return json.dumps(rec, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1, help="Interval alpha (e.g., 0.1 for 90% PI)")
    ap.add_argument("--cp_tol", type=int, default=10)
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows for faster sweeps")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    ap.add_argument("--smoke", type=int, default=0, help="Run only the first N combos (quick test)")
    ap.add_argument("--out", type=str, default="-", help="NDJSON path or '-' for stdout")
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    ap.set_defaults(progress=True)
    args = ap.parse_args()

    data_path = _resolve_data_path(args.data)
    combos: list[dict[str, float | int]] = list(_combo_iter())
    if args.smoke > 0:
        combos = combos[: args.smoke]
    total = len(combos)

    # output
    if args.out == "-" or not args.out:
        fh = sys.stdout
        close_fh = False
    else:
        fh = open(args.out, "w", buffering=1, encoding="utf-8")
        close_fh = True

    # progress impl: per combo
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # noqa: N816

    done = 0
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [ex.submit(_run_one, data_path, args.alpha, args.cp_tol, args.max_rows, c) for c in combos]

            if args.progress and tqdm is not None:
                pbar = tqdm(total=total, unit="combo", smoothing=0.05)
                for fut in as_completed(futs):
                    try:
                        line = fut.result()
                        fh.write(line + "\n")
                    except Exception as e:
                        sys.stderr.write(f"\n[sweep] combo failed: {e}\n")
                        sys.stderr.flush()
                    finally:
                        done += 1
                        pbar.update(1)
                pbar.close()
            else:
                if args.progress:
                    sys.stderr.write(f"Progress: 0/{total}\r")
                    sys.stderr.flush()
                for fut in as_completed(futs):
                    try:
                        line = fut.result()
                        fh.write(line + "\n")
                    except Exception as e:
                        sys.stderr.write(f"\n[sweep] combo failed: {e}\n")
                        sys.stderr.flush()
                    finally:
                        done += 1
                        if args.progress:
                            sys.stderr.write(f"Progress: {done}/{total}\r")
                            sys.stderr.flush()
                if args.progress:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
    finally:
        if close_fh:
            fh.close()


if __name__ == "__main__":
    main()
