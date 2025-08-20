#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
URL="${URL:-http://127.0.0.1:${PORT}/predict}"
TOKEN="${SERVICE_TOKEN:-}"
N="${N:-20000}"         # total requests per run
CONC_LIST="${CONC_LIST:-1 2 4 8 16}"
OUTDIR="${OUTDIR:-artifacts/bench}"
PIDFILE="${PIDFILE:-/tmp/bench_server.pid}"
START_SERVER="${START_SERVER:-0}"   # 1 => script starts/stops uvicorn
UVLOOP="${UVLOOP:-0}"              # 1 => use uvloop+httptools
WORKERS="${WORKERS:-1}"

mkdir -p "$OUTDIR/raw" "$OUTDIR/probe"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 1; }; }
need python
need awk
need ps
need sed
need date

# 'hey' is required
need hey 

start_server() {
  if [ "$START_SERVER" != "1" ]; then return; fi
  local loop_args=()
  if [ "$UVLOOP" = "1" ]; then
    loop_args+=(--loop uvloop --http httptools)
  fi
  UVICORN_CMD=(uvicorn service.app:app --port "$PORT" --workers "$WORKERS" "${loop_args[@]}")
  echo "Starting: ${UVICORN_CMD[*]}"
  "${UVICORN_CMD[@]}" & echo $! > "$PIDFILE"
  sleep 1
}

stop_server() {
  if [ "$START_SERVER" != "1" ]; then return; fi
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}

bearer() {
  if [ -n "$TOKEN" ]; then echo "-H" "authorization: Bearer $TOKEN"; fi
}

run_probe() {
  local pid="$1" outfile="$2"
  (
    echo "#ts_s pid rss_kb pcpu"
    while kill -0 "$pid" 2>/dev/null; do
      TS=$(date +%s)
      ps -o pid= -o rss= -o pcpu= -p "$pid" | awk -v ts="$TS" '{print ts, $1, $2, $3}'
      sleep 0.2
    done
  ) > "$outfile" &
  echo $!
}

make_body_file() {
  local out="$1"
  local rand
  rand=$(LC_ALL=C tr -dc a-z0-9 </dev/urandom | head -c 6)
  cat > "$out" <<JSON
{
  "series_id":"bench_${rand}",
  "timestamp":"2024-01-02T10:00:00Z",
  "target_timestamp":"2024-01-02T11:00:00Z",
  "x":0.001,
  "covariates":{"rv":0.02,"ewm_vol":0.015,"ac1":0.1,"z":0.0}
}
JSON
}

run_one() {
  local conc="$1"
  local raw="$OUTDIR/raw/hey_c${conc}.txt"
  local csv="$OUTDIR/raw/hey_c${conc}.csv"
  local probe="$OUTDIR/probe/proc_c${conc}.txt"

  # body into a temp file; hey reads with -D (data-file)
  local REQ_BODY
  REQ_BODY="$(mktemp)"
  make_body_file "$REQ_BODY"

  # find server pid to sample RSS/CPU
  local spid=""
  if [ -f "$PIDFILE" ]; then spid="$(cat "$PIDFILE")"
  else spid="$(pgrep -f 'uvicorn service.app:app' | head -n1 || true)"
  fi
  local probe_pid=""
  if [ -n "$spid" ]; then probe_pid=$(run_probe "$spid" "$probe"); fi

  # Fixed-request run so hey prints p50/p95/p99; use -D to send file body
  hey -n "$N" -c "$conc" -m POST $(bearer) -H 'content-type: application/json' -D "$REQ_BODY" "$URL" \
    | tee "$raw"

  # Optional CSV (not all hey builds support -o csv)
  hey -n "$N" -c "$conc" -m POST $(bearer) -H 'content-type: application/json' -D "$REQ_BODY" -o csv "$URL" > "$csv" || true

  rm -f "$REQ_BODY" || true
  if [ -n "$probe_pid" ]; then kill "$probe_pid" 2>/dev/null || true; fi
}

start_server

echo "URL=$URL"
for C in $CONC_LIST; do
  echo "== Concurrency $C =="
  run_one "$C"
done

stop_server

# Summarize to JSON
python - <<'PY'
import json, re, os
from pathlib import Path

outdir = Path(os.getenv("OUTDIR","artifacts/bench"))
rows = []
for raw in sorted((outdir/"raw").glob("hey_c*.txt")):
    m = re.search(r"hey_c(\d+)\.txt$", raw.name)
    if not m:
        continue
    conc = int(m.group(1))
    txt = raw.read_text()

    rps = float(re.search(r"Requests/sec:\s+([0-9\.]+)", txt).group(1)) if re.search(r"Requests/sec:", txt) else float("nan")

    def pct(p):
        mm = re.search(rf" +{p}%\s+([0-9\.]+)(\w+)", txt)
        if not mm: return float("nan")
        val, unit = float(mm.group(1)), mm.group(2).lower()
        if unit.startswith("ms"): return val/1000.0
        if unit.startswith("us"): return val/1_000_000.0
        return val  # seconds
    rows.append({
        "concurrency": conc,
        "rps": rps,
        "p50_s": pct(50),
        "p95_s": pct(95),
        "p99_s": pct(99),
        "raw": raw.name
    })

(outdir/"summary.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
