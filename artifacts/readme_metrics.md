### Backtest — Simulated Data

- Points: **2999**
- MAE: **0.0213**
- RMSE: **0.0305**
- sMAPE: **66.42%**
- Coverage (1-α): **0.911**
- Latency p50/p95: **1.1 / 1.2 ms**
- Change-point metrics:
  - Precision: **1.000**
  - Recall: **0.042**
  - Mean delay (ticks): **2.00**
  - p95 delay (ticks): **2**
  - Chatter (/1000): **0.3**
  - False alarm rate: **0.000**

![Simulated Backtest](artifacts/plot_sim.png)

### Service Smoke Test

Example `/predict` ➜ `/truth` roundtrip (shortened):

- `prediction_id`: `23a2acff-d499-4acd-918b-b46480a35de4`
- regime: `low_vol`  | score: `0.004`
- truth status: `ok`  | idempotent: `False`

