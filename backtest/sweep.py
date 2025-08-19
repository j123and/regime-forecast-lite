from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _to_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return math.nan


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
    """Read the replay once into memory so we can reuse across grid points."""
    return list(Replay(path, covar_cols=["rv", "ewm_vol", "ac1", "z"]))


def _iter_rows(rows: list[dict[str, Any]], max_rows: int | None) -> Iterable[dict[str, Any]]:
    if max_rows is None or max_rows >= len(rows):
        return rows
    return rows[:max_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cp_tol", type=int, default=10)
    ap.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows for faster sweeps")
    args = ap.parse_args()

    data_path = _resolve_data_path(args.data)

    base = load_config()

    hazards = [1 / 500, 1 / 300, 1 / 200, 1 / 120, 1 / 80]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    cooldowns = [3, 5, 10]
    router_switch_thresholds = [0.0, 0.1, 0.2]
    router_penalties = [0.0, 0.05, 0.1]
    freeze_ticks = [0, 3, 5]

    # BIG SPEED-UP: read once
    rows = _materialize(data_path)

    for h in hazards:
        for thr in thresholds:
            for cd in cooldowns:
                for rst in router_switch_thresholds:
                    for pen in router_penalties:
                        for frz in freeze_ticks:
                            cfg = copy.deepcopy(base)
                            d = cfg.setdefault("detector", {})
                            d["hazard"] = float(h)
                            d["threshold"] = float(thr)
                            d["cooldown"] = int(cd)

                            r = cfg.setdefault("router", {})
                            r["switch_threshold"] = float(rst)
                            r["switch_penalty"] = float(pen)
                            r["freeze_on_recent_cp"] = bool(frz > 0)
                            r["freeze_ticks"] = int(frz)

                            pipe = Pipeline(cfg)
                            runner = BacktestRunner(
                                alpha=args.alpha,
                                cp_tol=args.cp_tol,
                                cp_threshold=thr,
                                cp_cooldown=cd,
                            )
                            stream = _iter_rows(rows, args.max_rows)
                            metrics, _ = runner.run(pipe, stream)

                            out: dict[str, float | int] = {
                                "hazard": float(h),
                                "threshold": float(thr),
                                "cooldown": int(cd),
                                "router_switch_threshold": float(rst),
                                "router_penalty": float(pen),
                                "freeze_ticks": int(frz),
                            }
                            for k in [
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
                            ]:
                                out[k] = _to_float(metrics.get(k))
                            print(json.dumps(out))


if __name__ == "__main__":
    main()
