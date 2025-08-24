import argparse
import asyncio
from datetime import UTC, datetime, timedelta

import httpx


def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)

async def run(base_url, warmup, samples, step_seconds):
    async with httpx.AsyncClient(timeout=5.0) as client:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        series = "bench"
        x = 0.001
        cov = {"rv": 0.01, "ewm_vol": 0.012, "ac1": 0.1, "z": -0.2}

        # warmup
        for i in range(warmup):
            ts = t0 + timedelta(seconds=i * step_seconds)
            payload = {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "x": x,
                "covariates": cov,
                "series_id": series,
                "target_timestamp": (ts + timedelta(seconds=step_seconds)).isoformat().replace("+00:00", "Z"),
            }
            r = await client.post(f"{base_url}/predict", json=payload)
            r.raise_for_status()

        # measured
        lat = []
        for i in range(samples):
            ts = t0 + timedelta(seconds=(warmup + i) * step_seconds)
            payload = {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "x": x,
                "covariates": cov,
                "series_id": series,
                "target_timestamp": (ts + timedelta(seconds=step_seconds)).isoformat().replace("+00:00", "Z"),
            }
            r = await client.post(f"{base_url}/predict", json=payload)
            r.raise_for_status()
            service_ms = r.json().get("latency_ms", {}).get("service_ms")
            if service_ms is not None:
                lat.append(float(service_ms))

        p50 = percentile(lat, 0.50)
        p95 = percentile(lat, 0.95)
        print(f"Samples: {len(lat)}  (warmup: {warmup})")
        print(f"p50 service_ms: {p50:.3f}")
        print(f"p95 service_ms: {p95:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--step-seconds", type=int, default=3600)  # 1h steps for timestamps
    args = ap.parse_args()
    asyncio.run(run(args.url, args.warmup, args.samples, args.step_seconds))
