from __future__ import annotations
import argparse
import json

from core.config import load_config
from core.pipeline import Pipeline
from data.replay import Replay
from backtest.runner import BacktestRunner

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config()
    pipe = Pipeline(cfg=cfg)
    stream = Replay(args.data)
    bt = BacktestRunner(alpha=args.alpha)
    metrics, log = bt.run(pipe, stream)
    print(json.dumps({"metrics": metrics, "n_points": len(log)}, indent=2))

if __name__ == "__main__":
    main()
