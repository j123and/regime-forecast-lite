from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta

import pandas as pd


def simulate(
    n: int,
    seg_mean_scale: float = 0.015,
    seg_vol_low: float = 0.01,
    seg_vol_high: float = 0.04,
    p_vol_switch: float = 0.25,
    seg_len_min: int = 50,
    seg_len_max: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Piecewise Gaussian: each segment has constant mean and vol.
    cp=1 on the LAST index of each segment (except very last) to match runner semantics.
    """
    rnd = random.Random(seed)
    ts: list[str] = []
    x: list[float] = []
    cp: list[int] = []

    t = datetime(2024, 1, 1, 0, 0)
    i = 0
    mu = 0.0
    vol = seg_vol_low
    while i < n:
        seg_len = rnd.randint(seg_len_min, seg_len_max)
        seg_len = min(seg_len, n - i)
        # new segment params
        mu += rnd.gauss(0.0, seg_mean_scale)
        if rnd.random() < p_vol_switch:
            vol = seg_vol_high if math.isclose(vol, seg_vol_low, rel_tol=0.0, abs_tol=1e-12) else seg_vol_low
        # generate
        for k in range(seg_len):
            ts.append((t + timedelta(hours=1)).isoformat() + "Z")
            t += timedelta(hours=1)
            x.append(rnd.gauss(mu, vol))
            # mark cp on last tick of current segment (except if it's the very end of series)
            is_last_tick_of_segment = (k == seg_len - 1) and (i + seg_len < n)
            cp.append(1 if is_last_tick_of_segment else 0)
        i += seg_len

    return pd.DataFrame({"timestamp": ts, "x": x, "cp": cp})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mean_scale", type=float, default=0.015)
    ap.add_argument("--vol_low", type=float, default=0.01)
    ap.add_argument("--vol_high", type=float, default=0.04)
    ap.add_argument("--p_vol_switch", type=float, default=0.25)
    ap.add_argument("--seg_min", type=int, default=80)
    ap.add_argument("--seg_max", type=int, default=180)
    args = ap.parse_args()

    df = simulate(
        n=args.n,
        seg_mean_scale=args.mean_scale,
        seg_vol_low=args.vol_low,
        seg_vol_high=args.vol_high,
        p_vol_switch=args.p_vol_switch,
        seg_len_min=args.seg_min,
        seg_len_max=args.seg_max,
        seed=args.seed,
    )
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
