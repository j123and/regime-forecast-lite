#!/usr/bin/env python3
import time, json, argparse, statistics, http.client

def post_json(host, port, path, payload):
    body = json.dumps(payload)
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    _ = resp.read()
    conn.close()
    return resp.status

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    payload = {
        "timestamp": "2024-01-01T00:00:00Z",
        "x": 0.0,
        "covariates": {},
        "series_id": "bench",
        "target_timestamp": "2024-01-01T00:00:01Z"
    }

    lat = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        code = post_json(args.host, args.port, "/predict", payload)
        t1 = time.perf_counter()
        if code == 200:
            lat.append((t1 - t0) * 1000.0)

    if lat:
        p50 = statistics.median(lat)
        p95 = sorted(lat)[int(0.95 * len(lat)) - 1]
        print(f"Latency service_ms: p50={p50:.2f} ms, p95={p95:.2f} ms, N={len(lat)}")
    else:
        print("No successes")
