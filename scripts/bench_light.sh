#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8000}"    # use 8000 unless you override
: "${N:=1000}"       # total requests
: "${C:=4}"          # concurrency (keep small on WSL)
AUTH=()
if [[ -n "${SERVICE_TOKEN:-}" ]]; then
  AUTH=(-H "authorization: Bearer ${SERVICE_TOKEN}")
fi
BODY='{"timestamp":"2024-01-02T10:00:00Z","x":0.001,"covariates":{"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}}'

# 1) start server (single worker; stateful)
uvicorn service.app:app --port "$PORT" --workers 1 >/dev/null 2>&1 &
PID=$!
trap 'kill -TERM $PID >/dev/null 2>&1 || true' EXIT

# 2) wait for /healthz
for i in {1..100}; do
  curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null && break || sleep 0.1
done

# 3) warm up a bit
for _ in $(seq 1 50); do
  curl -s -o /dev/null -X POST "http://127.0.0.1:${PORT}/predict" \
    -H 'content-type: application/json' "${AUTH[@]}" -d "$BODY" || true
done

# 4) run N requests at concurrency C, capture per-request times (seconds)
TMP=$(mktemp)
export PORT BODY
start_ns=$(date +%s%N)
seq "$N" | xargs -n1 -P "$C" -I{} sh -c '
  curl -s -o /dev/null -w "%{time_total}\n" \
    -X POST "http://127.0.0.1:${PORT}/predict" \
    -H "content-type: application/json" '"${AUTH:+-H \"authorization: Bearer ${SERVICE_TOKEN}\"}"' \
    -d "$BODY" || echo "NaN"
' > "$TMP"
end_ns=$(date +%s%N)

# 5) summarize
# filter NaNs just in case
mapfile -t times < <(grep -E '^[0-9]+(\.[0-9]+)?$' "$TMP")
count=${#times[@]}
elapsed=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.6f", (e-s)/1e9}')
qps=$(awk -v n="$count" -v t="$elapsed" 'BEGIN{if(t>0) printf "%.1f", n/t; else print 0}')

# compute p50/p95 with Python (present everywhere)
pcts=$(python - "$TMP" <<'PY'
import sys
xs=[float(x.strip()) for x in open(sys.argv[1]) if x.strip().replace('.','',1).isdigit()]
xs.sort()
def pct(p):
    if not xs: return float('nan')
    k=(len(xs)-1)*p
    i=int(k); j=min(i+1,len(xs)-1); f=k-i
    return xs[i]*(1-f)+xs[j]*f
print(f"{pct(0.5)} {pct(0.95)}")
PY
)
p50=$(echo "$pcts" | awk '{print $1}')
p95=$(echo "$pcts" | awk '{print $2}')

echo "requests_ok: $count / $N"
echo "elapsed_s:   $elapsed"
echo "req_per_s:   $qps"
echo "p50_s:       $p50"
echo "p95_s:       $p95"

# 6) memory (RSS)
if kill -0 "$PID" 2>/dev/null; then
  echo -n "rss_mb:      "
  ps -o rss= -p "$PID" | awk '{printf "%.1f\n", $1/1024}'
fi
