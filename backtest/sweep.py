from __future__ import annotations

import argparse
import copy
import json
import math
from typing import Any

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _to_float(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cp_tol", type=int, default=10)
    args = ap.parse_args()

    base = load_config()

    hazards = [1 / 500, 1 / 300, 1 / 200, 1 / 120, 1 / 80]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    cooldowns = [3, 5, 10]
    router_switch_thresholds = [0.0, 0.1, 0.2]
    router_penalties = [0.0, 0.05, 0.1]
    freeze_ticks = [0, 3, 5]

    # IMPORTANT: recreate the stream per combination, otherwise you'll exhaust it
    for h in hazards:
        for thr in thresholds:
            for cd in cooldowns:
                for rst in router_switch_thresholds:
                    for pen in router_penalties:
                        for frz in freeze_ticks:
                            # new Replay for each run
                            stream = Replay(args.data, covar_cols=["rv", "ewm_vol", "ac1", "z"])

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
                            runner = BacktestRunner(alpha=args.alpha, cp_tol=args.cp_tol)
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
