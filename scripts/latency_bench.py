import argparse, asyncio, os, time
from datetime import UTC, datetime, timedelta
import httpx

def percentile(vals, p):
    if not vals: return float("nan")
    s = sorted(vals); k = (len(s)-1)*p; f = int(k); c = min(f+1, len(s)-1)
    return s[f] if f == c else s[f]*(c-k)+s[c]*(k-f)

async def run(base_url, warmup, samples, step_seconds, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key: headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
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
            await client.post(f"{base_url}/predict", json=payload)

        svc_ms = []
        e2e_ms = []
        truth_ms = []

        for i in range(samples):
            ts = t0 + timedelta(seconds=(warmup + i) * step_seconds)
            payload = {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "x": x,
                "covariates": cov,
                "series_id": series,
                "target_timestamp": (ts + timedelta(seconds=step_seconds)).isoformat().replace("+00:00", "Z"),
            }

            t1 = time.perf_counter_ns()
            r = await client.post(f"{base_url}/predict", json=payload)
            t2 = time.perf_counter_ns()
            r.raise_for_status()
            j = r.json()
            s = j.get("latency_ms", {}).get("service_ms")
            if s is not None:
                svc_ms.append(float(s))
            e2e_ms.append((t2 - t1) / 1e6)

            pid = j.get("prediction_id")
            if pid:
                t3 = time.perf_counter_ns()
                rt = await client.post(f"{base_url}/truth", json={"prediction_id": pid, "y_true": x})
                t4 = time.perf_counter_ns()
                rt.raise_for_status()
                truth_ms.append((t4 - t3) / 1e6)

        print(f"Samples: {len(e2e_ms)} (warmup: {warmup})")
        if svc_ms:
            print(f"service_ms   p50={percentile(svc_ms,0.50):.3f}  p95={percentile(svc_ms,0.95):.3f}")
        print(f"/predict E2E p50={percentile(e2e_ms,0.50):.3f}  p95={percentile(e2e_ms,0.95):.3f}")
        if truth_ms:
            print(f"/truth   E2E p50={percentile(truth_ms,0.50):.3f}  p95={percentile(truth_ms,0.95):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--step-seconds", type=int, default=3600)
    ap.add_argument("--api-key", default=os.getenv("SERVICE_API_KEY", ""))
    args = ap.parse_args()
    asyncio.run(run(args.url, args.warmup, args.samples, args.step_seconds, args.api_key))
