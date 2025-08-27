# core/pipeline.py
from __future__ import annotations

from collections import OrderedDict, deque
from math import isnan
from typing import Any

from core.config import load_config
from core.features import FeatureExtractor
from core.types import Tick


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return 0.0 if isnan(f) else f
    except Exception:
        return default


def _percentile(sorted_list, q: float) -> float:
    if not sorted_list:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    idx = int((len(sorted_list) - 1) * q)
    return float(sorted_list[idx])


class Pipeline:
    """
    Single-series, online pipeline:
      - EWMA mean/std as a simple one-step-ahead forecaster
      - Conformal intervals from absolute residuals (global + per-regime)
      - Binary regimes by EWMA std threshold
      - Pending map for service /truth matching by prediction_id
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self.fx = FeatureExtractor(
            alpha=float(self.cfg.get("ewma_alpha", 0.1)),
            min_warmup=int(self.cfg.get("min_warmup", 20)),
        )

        # Conformal quantile
        self.q = float(self.cfg.get("conformal_q", 0.9))
        self.maxlen = int(self.cfg.get("conformal_maxlen", 2000))
        self.global_res: deque[float] = deque(maxlen=self.maxlen)
        self.regime_res: dict[str, deque[float]] = {
            "calm": deque(maxlen=self.maxlen),
            "volatile": deque(maxlen=self.maxlen),
        }

        # Service book-keeping for /predict to /truth correlation
        self.pending: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.pending_cap = int(self.cfg.get("pending_cap", 4096))

        # Regime threshold on EWMA std
        self.vol_th = float(self.cfg.get("regime_vol_threshold", 0.01))

        # Remember last prediction so backtests can call update_truth(y) without IDs
        self._last_y_hat: float | None = None
        self._last_regime: str | None = None

    #  service hooks 
    def register_prediction(self, pred_id: str, y_hat: float, regime: str) -> None:
        self.pending[pred_id] = (float(y_hat), str(regime))
        self.pending.move_to_end(pred_id)
        while len(self.pending) > self.pending_cap:
            self.pending.popitem(last=False)
    
    def evict_prediction(self, pred_id: str) -> None:
        """Best-effort local eviction for service-level pending eviction."""
        self.pending.pop(pred_id, None)

    def _learn_residual(self, y_true: float, y_hat: float, regime: str | None) -> None:
        """Update residual buffers (global + regime) with |y - y_hat|."""
        try:
            r = abs(float(y_true) - float(y_hat))
        except Exception:
            return
        self.global_res.append(r)
        if regime and regime in self.regime_res:
            self.regime_res[regime].append(r)

    def update_truth_by_id(self, pred_id: str, y_true: float) -> bool:
        tup = self.pending.pop(pred_id, None)
        if tup is None:
            return False
        y_hat, regime = tup
        self._learn_residual(y_true, y_hat, regime)
        return True

    # Backtests can call this without a prediction_id.
    def update_truth(self, y: float, prediction_id: str | None = None) -> None:
        """
        Ingest realized truth for the most recent prediction (or for a specific prediction_id).
        """
        if prediction_id is not None:
            # Defer to the ID-based path used by the service
            self.update_truth_by_id(str(prediction_id), float(y))
            return
        # Learn against the last prediction produced by process()
        if self._last_y_hat is None:
            return
        self._learn_residual(float(y), float(self._last_y_hat), self._last_regime)

    #  main step 
    def process(self, tick: Tick) -> dict[str, Any]:
        x = _safe_float(tick["x"])
        f = self.fx.update(x)
        mean = float(f["ewm_mean"])
        std = float(f["ewm_std"])
        warmup = bool(f["warmup"])

        # Forecast: next-tick mean proxy
        y_hat = mean

        # Regime by scale threshold
        regime = "volatile" if std >= self.vol_th else "calm"

        # Build interval radius from residual buffers (per-regime with global fallback)
        reg_buf = list(self.regime_res.get(regime, deque()))
        reg_buf.sort()
        glob = list(self.global_res)
        glob.sort()

        r_reg = _percentile(reg_buf, self.q) if reg_buf else 0.0
        r_glob = _percentile(glob, self.q) if glob else 0.0

        degraded = False
        if len(reg_buf) < 30 and r_glob > r_reg:
            r = r_glob
            degraded = True
        else:
            r = max(r_reg, r_glob)

        interval_low = y_hat - r
        interval_high = y_hat + r

        # Detector score proxy from volatility; [0, 1] clipped
        score = min(max(std / max(self.vol_th, 1e-12), 0.0), 1.0)

        # Remember last prediction so update_truth() can learn next tick
        self._last_y_hat = float(y_hat)
        self._last_regime = str(regime)

        # Keep both explicit bounds and an intervals map for compatibility
        intervals = {
            f"alpha={1.0 - self.q:.2f}": [interval_low, interval_high],
            str(int(self.q * 100)): [interval_low, interval_high],  # legacy key
        }

        return {
            "y_hat": y_hat,
            "interval_low": interval_low,
            "interval_high": interval_high,
            "intervals": intervals,
            "regime": regime,
            "score": score,
            "warmup": warmup,
            "degraded": degraded,
        }

    #  snapshot state (buffers + pending only) 
    def state_dict(self) -> dict[str, Any]:
        return {
            "global_res": list(self.global_res),
            "regime_res": {k: list(v) for k, v in self.regime_res.items()},
            "pending": [
                {"prediction_id": pid, "y_hat": yh, "regime": rg}
                for pid, (yh, rg) in self.pending.items()
            ],
        }

    @classmethod
    def from_state(cls, cfg: dict[str, Any] | None, state: dict[str, Any]) -> Pipeline:
        self = cls(cfg)
        for v in state.get("global_res", []):
            self.global_res.append(float(v))
        for k in ("calm", "volatile"):
            for v in state.get("regime_res", {}).get(k, []):
                self.regime_res[k].append(float(v))
        for rec in state.get("pending", []):
            pid = rec.get("prediction_id")
            if pid:
                self.register_prediction(
                    str(pid),
                    float(rec.get("y_hat", 0.0)),
                    str(rec.get("regime", "calm")),
                )
        return self
