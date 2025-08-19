from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtest.runner import BacktestRunner
from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay


def _resolve_data_path(p: str) -> str:
    cand = Path(p)
    if cand.exists():
        return str(cand)
    base = Path(p).name
    for guess in (Path("data") / base,):
        if guess.exists():
            return str(guess)
    raise FileNotFoundError(f"Data file not found: '{p}' (also tried 'data/{base}')")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cp_tol", type=int, default=10)
    ap.add_argument(
        "--profile",
        choices=["sim", "market"],
        help="Load config/profiles/<profile>.yaml",
    )
    ap.add_argument(
        "--config",
        help="Path to a YAML config (overrides default/profile).",
    )
    args = ap.parse_args()

    data_path = _resolve_data_path(args.data)

    cfg = load_config(path=args.config, profile=args.profile)
    pipe = Pipeline(cfg)
    runner = BacktestRunner(alpha=args.alpha, cp_tol=args.cp_tol)

    stream = Replay(data_path, covar_cols=["rv", "ewm_vol", "ac1", "z"])
    metrics, log = runner.run(pipe, stream)

    out = {
        "metrics": metrics,
        "n_points": len(log),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
