#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np

def ar1_ma(y, window=200):
    y = pd.Series(y).astype(float).values
    n = len(y)
    y_pred = np.full(n, np.nan)
    for t in range(1, n):
        lo = max(0, t - window)
        x = y[lo:t-1]         # regressors: up to t-2
        z = y[lo+1:t]         # targets:   up to t-1
        if len(x) < 10:
            continue
        denom = float(x @ x) + 1e-12
        phi = float((x @ z) / denom)
        y_pred[t] = phi * y[t-1]
    return y_pred

def rw(y):
    y = np.asarray(y, float)
    y_pred = np.roll(y, 1)
    y_pred[0] = np.nan
    return y_pred

def ewma(y, alpha=0.2):
    y = np.asarray(y, float)
    y_pred = np.full_like(y, np.nan)
    m = 0.0
    have = False
    for t in range(len(y)):
        if have:
            m = alpha * y[t-1] + (1 - alpha) * m
            y_pred[t] = m
        else:
            if t >= 1:
                m = y[t-1]
                have = True
                y_pred[t] = m
    return y_pred

def metrics(y, yhat):
    mask = ~np.isnan(yhat)
    e = y[mask] - yhat[mask]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    return mae, rmse, int(mask.sum())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--col", default="x")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--ar1_win", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.col].astype(float).values

    for name, yhat in [
        ("RW", rw(y)),
        ("AR1", ar1_ma(y, window=args.ar1_win)),
        (f"EWMA(alpha={args.alpha})", ewma(y, alpha=args.alpha)),
    ]:
        mae, rmse, n = metrics(y, yhat)
        print(f"{name}: MAE={mae:.6f} RMSE={rmse:.6f} (N={n})")
