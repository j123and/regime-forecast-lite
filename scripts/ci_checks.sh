#!/usr/bin/env bash
set -euo pipefail

# 1) global coverage ±0.05
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import sys
cfg=load_config(profile="market"); m,_=BacktestRunner(alpha=0.1,cp_tol=10).run(
  Pipeline(cfg), Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
cov=float(m["coverage"]); print({"coverage":cov}); sys.exit(0 if abs(cov-0.9)<=0.05 else 1)
PY

# 2) rolling coverage (200) ≥0.85
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import pandas as pd, sys
_,log=BacktestRunner(alpha=0.1,cp_tol=10).run(Pipeline(load_config(profile="market")),
  Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
d=pd.DataFrame(log).dropna(subset=["y","ql","qh"]); d["hit"]=(d["y"]>=d["ql"])&(d["y"]<=d["qh"])
r=d["hit"].rolling(200,min_periods=200).mean().iloc[-1]; print({"roll200":float(r)}); sys.exit(0 if r>=0.85 else 1)
PY

# 3) residual alignment: conformal q90 within ±25% of backtest q90
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import pandas as pd, numpy as np, sys
from core.conformal import _weighted_quantile
m,log=BacktestRunner(alpha=0.1,cp_tol=10).run(Pipeline(load_config(profile="market")),
  Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
d=pd.DataFrame(log).dropna(subset=["y","y_hat"]); q_df=float(np.quantile((d["y"]-d["y_hat"]).abs(),0.9))
from core.pipeline import Pipeline as P; p=P(load_config(profile="market"))  # new to get conf
q_buf=float(_weighted_quantile(list(p.conf.res_global), list(p.conf.wts_global),0.9)) if p.conf.res_global else q_df
ratio=q_buf/q_df if q_df>0 else 1.0; print({"ratio":ratio}); sys.exit(0 if 0.8<=ratio<=1.25 else 1)
PY

# 4) miss symmetry
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import pandas as pd, sys
_,log=BacktestRunner(alpha=0.1,cp_tol=10).run(Pipeline(load_config(profile="market")),
  Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
d=pd.DataFrame(log).dropna(subset=["y","ql","qh"]); mu=int((d["y"]>d["qh"]).sum()); md=int((d["y"]<d["ql"]).sum())
tot=max(1,mu+md); frac=mu/tot; print({"frac_up":frac}); sys.exit(0 if 0.35<=frac<=0.65 else 1)
PY

# 5) latency budget
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import sys
m,_=BacktestRunner(alpha=0.1,cp_tol=10).run(Pipeline(load_config(profile="market")),
  Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
p50=float(m["latency_p50_ms"]); p95=float(m["latency_p95_ms"]); print({"p50":p50,"p95":p95})
sys.exit(0 if (p50<=12.0 and p95<=25.0) else 1)
PY

# 6) per-regime coverage ≥0.85 (if regimes exist)
python - <<'PY'
from backtest.runner import BacktestRunner; from core.config import load_config
from core.pipeline import Pipeline; from data.replay import Replay; import pandas as pd, sys
_,log=BacktestRunner(alpha=0.1,cp_tol=10).run(Pipeline(load_config(profile="market")),
  Replay("data/aapl_1h_logret.csv", covar_cols=["rv","ewm_vol","ac1","z"]))
d=pd.DataFrame(log); 
if "regime" not in d.columns: print({"skip":"no regime"}); sys.exit(0)
d=d.dropna(subset=["y","ql","qh","regime"]).copy(); d["hit"]=(d["y"]>=d["ql"])&(d["y"]<=d["qh"])
cov=d.groupby(d["regime"].astype(str))["hit"].mean().to_dict(); print({"cov_by_regime":cov})
sys.exit(0 if all(v>=0.85 for v in cov.values()) else 1)
PY
