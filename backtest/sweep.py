from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


# ---------- metrics we care about ----------
METRIC_KEYS: Tuple[str, ...] = (
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

# ---------- cached base config per process ----------
_BASE_CONFIG = None  # lazily loaded in each worker process


def _get_base_config() -> dict:
    global _BASE_CONFIG
    if _BASE_CONFIG is None:
        _BASE_CONFIG = load_config()
    return _BASE_CONFIG


def _to_json_safe_float(v: Any) -> float | None:
    """Return a JSON-friendly float or None (→ null) if not coercible/finite."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _combo_iter() -> Iterable[Dict[str, float | int]]:
    hazards = [1 / 500, 1 / 300, 1 / 200, 1 / 120, 1 / 80]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    cooldowns = [3, 5, 10]
    router_switch_thresholds = [0.0, 0.1, 0.2]
    router_penalties = [0.0, 0.05, 0.1]
    freeze_ticks = [0, 3, 5]

    for h, thr, cd, rst, pen, frz in itertools.product(
        hazards,
        thresholds,
        cooldowns,
        router_switch_thresholds,
        router_penalties,
        freeze_ticks,
    ):
        yield {
            "hazard": float(h),
            "threshold": float(thr),
            "cooldown": int(cd),
            "router_switch_threshold": float(rst),
            "router_penalty": float(pen),
            "freeze_ticks": int(frz),
        }


def _run_single(
    data_path: str,
    alpha: float,
    cp_tol: int,
    combo: Dict[str, float | int],
) -> str:
    """
    Worker: builds Replay/Pipeline/Runner and returns one NDJSON line.
    Recreates Replay for each run because the stream is consumable.
    """
    # Build config from cached base
    cfg = copy.deepcopy(_get_base_config())

    d = cfg.setdefault("detector", {})
    d["hazard"] = float(combo["hazard"])
    d["threshold"] = float(combo["threshold"])
    d["cooldown"] = int(combo["cooldown"])

    r = cfg.setdefault("router", {})
    r["switch_threshold"] = float(combo["router_switch_threshold"])
    r["switch_penalty"] = float(combo["router_penalty"])
    r["freeze_ticks"] = int(combo["freeze_ticks"])
    r["freeze_on_recent_cp"] = bool(int(combo["freeze_ticks"]) > 0)

    # New replay each time (avoid exhausting the stream)
    stream = Replay(data_path, covar_cols=["rv", "ewm_vol", "ac1", "z"])

    pipe = Pipeline(cfg)
    runner = BacktestRunner(alpha=alpha, cp_tol=cp_tol)
    metrics, _ = runner.run(pipe, stream)

    out: Dict[str, Any] = dict(combo)  # start with hyperparams
    for k in METRIC_KEYS:
        out[k] = _to_json_safe_float(metrics.get(k))

    # Strict JSON (nulls instead of NaN), compact separators for throughput
    return json.dumps(out, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cp_tol", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="Number of parallel workers (default: CPU count)")
    ap.add_argument("--out", type=str, default="-",
                    help="Output NDJSON file path or '-' for stdout")
    args = ap.parse_args()

    combos: List[Dict[str, float | int]] = list(_combo_iter())

    # Output target
    if args.out == "-" or not args.out:
        fh = sys.stdout
        close_fh = False
    else:
        fh = open(args.out, "w", buffering=1, encoding="utf-8")  # line-buffered
        close_fh = True

    try:
        # Parallel fan-out, stream results as they complete
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futures = [
                ex.submit(_run_single, args.data, args.alpha, args.cp_tol, combo)
                for combo in combos
            ]
            for fut in as_completed(futures):
                line = fut.result()  # will raise if worker failed
                fh.write(line + "\n")
                fh.flush()
    finally:
        if close_fh:
            fh.close()


if __name__ == "__main__":
    main()
