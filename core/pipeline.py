from __future__ import annotations

import time
from collections import OrderedDict
from typing import cast

from models.ewma import EWMAModel

from .conformal import OnlineConformal
from .detect.bocpd import BOCPD
from .features import FeatureExtractor
from .types import OnlineModel, Tick


class Pipeline:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}

        # features (keep minimal; no peeking inside FeatureExtractor)
        fe_cfg = self.cfg.get("features", {})
        self.fe = FeatureExtractor(
            **{k: v for k, v in fe_cfg.items() if k in {"win", "rv_win", "ewm_alpha"}}
        )

        # single detector (BOCPD is fine here; tests already exist)
        dcfg = self.cfg.get("detector", {})
        self.det = BOCPD(
            threshold=dcfg.get("threshold", 0.6),
            cooldown=dcfg.get("cooldown", 5),
            hazard=dcfg.get("hazard", 1 / 200),
            rmax=dcfg.get("rmax", 400),
            mu0=dcfg.get("mu0", 0.0),
            kappa0=dcfg.get("kappa0", 1e-3),
            alpha0=dcfg.get("alpha0", 1.0),
            beta0=dcfg.get("beta0", 1.0),
            vol_threshold=dcfg.get("vol_threshold", 0.02),
        )

        # single online model (EWMA)
        self.model: OnlineModel = cast(
            OnlineModel, EWMAModel(alpha=self.cfg.get("model_alpha", 0.2))
        )

        # conformal intervals (single by default)
        ccfg = self.cfg.get("conformal", {})
        self.alpha_main = float(ccfg.get("alpha_main", 0.1))
        self.alpha_list: list[float] = list(ccfg.get("alphas", [self.alpha_main]))
        self.conf = OnlineConformal(
            window=ccfg.get("window", 500),
            decay=ccfg.get("decay", 1.0),
            by_regime=ccfg.get("by_regime", False),  # simpler: global residuals by default
            cold_scale=ccfg.get("cold_scale", 0.01),
        )

        # pending predictions keyed by id (for /truth)
        self._pending: OrderedDict[str, dict] = OrderedDict()
        self._pending_cap: int = int(self.cfg.get("pending_cap", 4096))

        # NEW: last prediction for backtester (no ids needed)
        self._last_pending: dict[str, float | str] | None = None

    # ---------- public API used by the service ----------

    def register_prediction(self, pred_id: str, y_hat: float, regime_label: str | None) -> None:
        self._pending[pred_id] = {"y_hat": float(y_hat), "regime": regime_label}
        while len(self._pending) > self._pending_cap:
            self._pending.popitem(last=False)

    def update_truth_by_id(self, pred_id: str, y_true: float) -> bool:
        item = self._pending.pop(pred_id, None)
        if item is None:
            return False
        self.conf.update(float(item["y_hat"]), float(y_true), item.get("regime"))
        return True

    def pending_count(self) -> int:
        return len(self._pending)

    # ---------- NEW: simple updater for the backtester ----------

    def update_truth(self, y_true: float) -> None:
        """
        Update conformal buffers using the most recent prediction produced by `process`.
        Safe no-op if called before any prediction.
        """
        if self._last_pending is None:
            return
        self.conf.update(
            float(self._last_pending.get("y_hat", 0.0)),
            float(y_true),
            str(self._last_pending.get("regime") or ""),
        )
        self._last_pending = None

    # ---------- main processing ----------

    def process(self, tick: Tick) -> dict:
        t0 = time.perf_counter()

        # 1) features
        feats = self.fe.update(tick)
        t1 = time.perf_counter()

        # 2) detector
        det = self.det.update(float(tick["x"]), feats)
        t2 = time.perf_counter()

        # 3) model predict+update
        y_hat, _ = self.model.predict_update(tick, feats)
        t3 = time.perf_counter()

        # 4) intervals from current conformal state
        vol_hint = float(feats.get("ewm_vol") or feats.get("rv") or 0.01)
        ql, qh = self.conf.interval(
            y_hat, alpha=self.alpha_main, regime_label=det["regime_label"], scale_hint=vol_hint
        )
        multi = self.conf.interval(
            y_hat,
            regime_label=det["regime_label"],
            scale_hint=vol_hint,
            alphas_multi=self.alpha_list,
        )
        t4 = time.perf_counter()

        # remember for `update_truth`
        self._last_pending = {"y_hat": float(y_hat), "regime": str(det["regime_label"])}

        lat = {
            "features_ms": (t1 - t0) * 1000.0,
            "detector_ms": (t2 - t1) * 1000.0,
            "model_ms": (t3 - t2) * 1000.0,
            "conformal_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
        }

        return {
            "y_hat": float(y_hat),
            "interval_low": float(ql),
            "interval_high": float(qh),
            "regime": det["regime_label"],
            "score": float(det["regime_score"]),
            "latency_ms": lat,
            "warmup": len(self.conf.res_global) < 30,
            "intervals": multi if isinstance(multi, dict) else {},
        }
