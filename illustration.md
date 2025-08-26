here’s a clean, text-only map of how everything fits together — both for offline backtests and the live API. no tables, just arrows.

```
DATA SOURCES
============
[ data/sim_cp.py ]  ──(CSV: timestamp,x,cp?)────────┐
[ data/yahoo_fetch.py ] ──(CSV: timestamp,x)────────┤
(you can also bring your own CSV/Parquet)           └──▶

REPLAY → TICKS
==============
[ data/replay.py ]
  CSV/Parquet → yields ticks like:
    {"timestamp": str, "x": float, "covariates": {...}, optional "cp": 0/1}

OFFLINE BACKTEST FLOW
=====================
backtest.cli → BacktestRunner → Pipeline → Features → Intervals → Metrics → (optional) Plot

[ backtest/cli.py ]
  └─ loads cfg via core/config.py (default → profile overrides)
  └─ builds Pipeline(cfg)
  └─ stream = Replay(data, covar_cols=[...])
  └─ runs BacktestRunner(alpha, cp_tol).run(pipe, stream)
      │
      ▼
[ backtest/runner.py ]
  loop over ticks t=1..T:
    1) ingest previous truth y_{t-1} into pipe (leakage-safe)
    2) pred_t = _predict(pipe, tick_t)
       - measure compute latency
    3) evaluate pred_{t-1} vs y_t and log:
       y, y_hat, interval [ql,qh], regime, score/cp_prob, cp_true, latency
  after loop:
    - metrics:
        MAE/RMSE/SMAPE  → backtest/metrics.py
        Coverage        → fraction y∈[ql,qh]
        Latency p50/p95 → simple order stats
        CP metrics      → optional (threshold/cooldown + tolerance matching)
      returns (metrics, log)
      │
      ▼
[ scripts/plot_backtest.py ] (optional)
  - trims last N points
  - plots y, y_hat, shaded PI, vertical lines at cp_true
  - shades spans where regime == "volatile"

PIPELINE INTERNALS (used by both backtest + service)
====================================================
[ core/pipeline.py ]
  process(tick):
    1) FeatureExtractor.update(x)  → EWMA mean/std, warmup flag
    2) y_hat := EWMA mean (one-step-ahead proxy)
    3) regime := "volatile" if std ≥ cfg.regime_vol_threshold else "calm"
    4) interval radius r:
         - quantile q = cfg.conformal_q (e.g., 0.90)
         - r_reg := percentile(|residuals| in current regime)
         - r_glob := percentile(|residuals| global)
         - r := max(r_reg, r_glob); if regime buf small, fallback to global (degraded=True)
       interval = [y_hat - r, y_hat + r]
    5) score := clip(std / vol_threshold, 0..1)  (simple volatility score)
    6) remember last y_hat/regime for id-less truth updates
    7) return dict: y_hat, interval_low/high, intervals map, regime, score, warmup, degraded

  register_prediction(pred_id, y_hat, regime):
    - stores (y_hat, regime) in FIFO pending (per series)

  update_truth_by_id(pred_id, y_true):
    - pop (y_hat, regime) from pending
    - learn residual |y_true - y_hat| into global & regime buffers

[ core/features.py ]
  - EWMA of mean (m_t) and second moment (s_t)
  - var = s_t - m_t^2 (floored at 0), std = sqrt(var)
  - warmup flag until min_warmup updates
  - returns: ewm_mean, ewm_var, ewm_std, z, ac1 (stub), rv (alias of var)

[ core/config.py ]
  - loads default → overlays profile (market/sim) unless an explicit file is provided
  - flattens detector.vol_threshold → regime_vol_threshold for the pipeline

LIVE SERVICE FLOW (HTTP)
========================
Client → FastAPI → Pipeline → Pending Index → Truth → Residual Buffers → Future Intervals

[ service/app.py ]
  /predict:
    - auth/rate limit (only enforced in specific tests or if SERVICE_API_KEY set)
    - get per-series Pipeline (LRU up to max_series)
    - tick := {"timestamp", "x", "covariates"}
    - pred := pipe.process(tick)
    - pred_id := uuid4
    - pipe.register_prediction(pred_id, pred.y_hat, pred.regime)
    - remember pending globally:
        _PENDING_BY_KEY[(series_id, target_ts)] = pred_id  (FIFO cap PENDING_CAP)
        _PID_TO_SERIES[pred_id] = series_id
        on cap hit, evict oldest (and best-effort pipe.evict_prediction)
    - return PredictOut (y_hat, interval, regime, score, latency_ms)

  /truth:
    - parse y from {y, y_true, value}
    - resolve prediction_id (direct OR by series_id+target_timestamp)
    - look up series via _PID_TO_SERIES; if missing/evicted → 404
    - idempotency: if already applied within TTL → return idempotent=True
    - pipe.update_truth_by_id(pred_id, y_true):
        * removes from pipe.pending
        * updates residual buffers (global + regime)
      mark applied; return status ok

  /metrics: Prometheus
  /healthz: trivial

[ service/middleware.py ]
  - adds X-Service-MS + Server-Timing headers
  - does NOT touch bodies (safe)

TESTS & CI
==========
[ tests/test_eviction_and_nan.py ]
  - sets PENDING_CAP=1, checks evicted pred returns 404 on /truth
  - sends x="NaN", expects 422

[ tests/test_rate_limit.py ]
  - RATE_LIMIT_PER_MINUTE=2 → third /predict hits 429
  - API_KEY=secret → missing header 401, correct header 200

[ scripts/ci_checks.sh ]
  - runs a synthetic dataset if none present
  - validates: global coverage ≈ target, rolling coverage ok, buffer vs log residuals aligned,
               miss symmetry balanced, latency budget OK, per-regime coverage OK

OPTIONAL / AUX
==============
[ core/detect/bocpd.py ]
  - “BOCPD-like” wrapper with z-score → cp_prob (API-compatible with older code)
  - not wired into Pipeline by default; kept as an alternate detector

[ core/conformal.py ]
  - a more featureful online conformal class (window + decay + per-regime + weighted quantiles)
  - pipeline currently uses a lighter built-in variant (deque + empirical percentile)
```

that’s the whole story: data files → `Replay` → `BacktestRunner` & `Pipeline` (offline), or HTTP `/predict` → `Pipeline` → `/truth` → residuals (online). config plugs in thresholds/alpha, and tests/CI pin the behavior.
