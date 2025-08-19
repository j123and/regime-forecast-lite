from __future__ import annotations

import time
from typing import cast

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
        self.fe = FeatureExtractor(**{k: v for k, v in fe_cfg.items() if k in {"win", "rv_win", "ewm_alpha"}})

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
            dwell_min=rcfg.get("dwell_min", 10),
            switch_threshold=rcfg.get("switch_threshold", 0.0),
            switch_penalty=rcfg.get("switch_penalty", 0.0),
            freeze_on_recent_cp=rcfg.get("freeze_on_recent_cp", False),
            freeze_ticks=rcfg.get("freeze_ticks", 0),
        )

        self.models: dict[str, OnlineModel] = {
            "arima": cast(OnlineModel, ARIMAModel()),
            "xgb": cast(OnlineModel, XGBModel()),
        }

        ccfg = self.cfg.get("conformal", {})
        self.alpha_main = float(ccfg.get("alpha_main", 0.1))
        self.alpha_list: list[float] = list(ccfg.get("alphas", [self.alpha_main]))
        self.conf = OnlineConformal(
            window=ccfg.get("window", 500),
            decay=ccfg.get("decay", 1.0),
            by_regime=ccfg.get("by_regime", True),
            cold_scale=ccfg.get("cold_scale", 0.01),
        )

        self._pending_truth_updates = 0
        self._last_y_hat: float | None = None
        self._last_regime: str | None = None

    def update_truth(self, y_true: float) -> None:
        if self._last_y_hat is not None:
            self.conf.update(self._last_y_hat, y_true, self._last_regime)
        self._pending_truth_updates = max(0, self._pending_truth_updates - 1)

    def process(self, tick: Tick) -> dict[str, float | str | dict[str, float] | bool | dict[str, tuple[float, float]]]:
        t0 = time.perf_counter()

        feats = self.fe.update(tick)
        t1 = time.perf_counter()

        det = self.det.update(float(tick["x"]), feats)
        t2 = time.perf_counter()

        model_name = self.router.choose(det["regime_label"], det["regime_score"], det.get("meta", {}))
        t3 = time.perf_counter()

        y_hat, _ = self.models[model_name].predict_update(tick, feats)
        t4 = time.perf_counter()

        vol_hint = float(feats.get("ewm_vol") or feats.get("rv") or 0.01)
        ql, qh = self.conf.interval(y_hat, alpha=self.alpha_main, regime_label=det["regime_label"], scale_hint=vol_hint)
        multi = self.conf.interval(
            y_hat, regime_label=det["regime_label"], scale_hint=vol_hint, alphas_multi=self.alpha_list
        )
        t5 = time.perf_counter()

        self._last_y_hat = float(y_hat)
        self._last_regime = det["regime_label"]
        self._pending_truth_updates += 1

        lat = {
            "features_ms": (t1 - t0) * 1000.0,
            "detector_ms": (t2 - t1) * 1000.0,
            "router_ms": (t3 - t2) * 1000.0,
            "model_ms": (t4 - t3) * 1000.0,
            "conformal_ms": (t5 - t4) * 1000.0,
            "total_ms": (t5 - t0) * 1000.0,
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
