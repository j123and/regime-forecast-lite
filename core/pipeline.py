from __future__ import annotations

import time
from typing import Any, cast

from models.arima import ARIMAModel
from models.xgb import XGBModel

from .conformal import OnlineConformal
from .detect.bocpd import BOCPD
from .features import FeatureExtractor
from .router import Router
from .types import OnlineModel, Tick


class Pipeline:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}

        fe_cfg = self.cfg.get("features", {})
        self.fe = FeatureExtractor(
            **{k: v for k, v in fe_cfg.items() if k in {"win", "rv_win", "ewm_alpha"}}
        )

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

        rcfg = self.cfg.get("router", {})
        self.router = Router(
            low_model=rcfg.get("low_model", "arima"),
            high_model=rcfg.get("high_model", "xgb"),
            dwell_min=rcfg.get("dwell_min", 10),
            switch_threshold=rcfg.get("switch_threshold", 0.0),
            switch_penalty=rcfg.get("switch_penalty", 0.0),
            freeze_on_recent_cp=rcfg.get("freeze_on_recent_cp", False),
            freeze_ticks=rcfg.get("freeze_ticks", 5),
            cp_spike_threshold=rcfg.get("cp_spike_threshold", 0.6),
        )

        mcfg = self.cfg.get("models", {})
        self.models: dict[str, OnlineModel] = {
            "arima": cast(OnlineModel, ARIMAModel(**mcfg.get("arima", {}))),
            "xgb": cast(OnlineModel, XGBModel(**mcfg.get("xgb", {}))),
        }

        ccfg = self.cfg.get("conformal", {})
        self.conf = OnlineConformal(
            window=ccfg.get("window", 500),
            decay=ccfg.get("decay", 1.0),
            cold_scale=ccfg.get("cold_scale", 0.0),
        )
        self._alpha = float(ccfg.get("alpha", 0.1))

        self._last_y_hat: float | None = None
        self._pending_truth_updates: int = 0
        self._warmup_min_res = int(ccfg.get("warmup_min_res", 30))

    def update_truth(self, y_true: float, y_hat: float | None = None) -> None:
        yh = float(y_hat) if y_hat is not None else (self._last_y_hat if self._last_y_hat is not None else None)
        if yh is None:
            return
        self.conf.update(yh, float(y_true))

    def process(self, tick: Tick) -> dict[str, Any]:
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        feats = self.fe.update(tick)
        t2 = time.perf_counter()

        det = self.det.update(tick["x"], feats)
        t3 = time.perf_counter()

        model_name = self.router.choose(
            det["regime_label"], det["regime_score"], det.get("meta", {})
        )
        t4 = time.perf_counter()

        y_hat, _ = self.models[model_name].predict_update(tick, feats)
        t5 = time.perf_counter()

        ql, qh = self.conf.interval(
            y_hat, alpha=self._alpha, scale_hint=float(feats.get("ewm_vol", 0.0))
        )
        self._last_y_hat = float(y_hat)
        self._pending_truth_updates += 1
        t6 = time.perf_counter()

        warmup = self.conf.n < self._warmup_min_res
        degraded = bool(det.get("meta", {}).get("error", False))

        return {
            "y_hat": float(y_hat),
            "interval_low": float(ql),
            "interval_high": float(qh),
            "regime": det["regime_label"],
            "score": float(det["regime_score"]),
            "latency_ms": {
                "features": (t2 - t1) * 1e3,
                "detector": (t3 - t2) * 1e3,
                "router": (t4 - t3) * 1e3,
                "model": (t5 - t4) * 1e3,
                "conformal": (t6 - t5) * 1e3,
                "total": (t6 - t0) * 1e3,
            },
            "warmup": warmup,
            "degraded": degraded,
        }
