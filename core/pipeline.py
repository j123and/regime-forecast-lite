# core/pipeline.py
from __future__ import annotations

import json
import time
from collections import OrderedDict, deque
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

        # --- alignment state (legacy FIFO path) ---
        self._yhat_prev: float | None = None
        self._regime_prev: str | None = None
        self._yhat_next: float | None = None
        self._regime_next: str | None = None

        # --- NEW: pending map for multi-outstanding predictions ---
        # pred_id -> {"y_hat": float, "regime": str|None}
        self._pending: "OrderedDict[str, dict]" = OrderedDict()
        self._pending_cap: int = int(self.cfg.get("pending_cap", 4096))

    # ---------- public API used by the service ----------

    def register_prediction(self, pred_id: str, y_hat: float, regime_label: str | None) -> None:
        """
        Register a newly emitted prediction keyed by pred_id so we can match
        a future /truth by id or pop FIFO for legacy.
        """
        self._pending[pred_id] = {"y_hat": float(y_hat), "regime": regime_label}
        # cap memory
        while len(self._pending) > self._pending_cap:
            self._pending.popitem(last=False)  # evict oldest

    def update_truth_by_id(self, pred_id: str, y_true: float) -> bool:
        """
        Apply a truth to a specific pending prediction by id.
        Returns True if applied, False if not found (already applied or evicted).
        """
        item = self._pending.pop(pred_id, None)
        if item is None:
            return False
        self.conf.update(float(item["y_hat"]), float(y_true), item.get("regime"))
        return True

    def pending_count(self) -> int:
        return len(self._pending)

    # ---------- legacy FIFO path (still supported) ----------

    def update_truth(self, y_true: float) -> None:
        """
        Legacy FIFO alignment: if we have pending ids, pop the oldest and apply.
        Otherwise, fall back to the previous-pointer scheme.
        """
        # Prefer proper pending map if available
        if self._pending:
            # pop oldest pred_id and apply
            _, item = self._pending.popitem(last=False)
            self.conf.update(float(item["y_hat"]), float(y_true), item.get("regime"))
            return

        # Fallback: pointer-based (compat for old clients/tests)
        if self._yhat_prev is not None:
            self.conf.update(self._yhat_prev, float(y_true), self._regime_prev)

        self._yhat_prev = self._yhat_next
        self._regime_prev = self._regime_next
        # _yhat_next/_regime_next will be overwritten on next process()

    # ---------- main processing ----------

    def process(self, tick: Tick) -> dict[str, float | str | dict[str, float] | bool | dict[str, tuple[float, float]]]:
        t0 = time.perf_counter()

        # 1) features
        feats = self.fe.update(tick)
        t1 = time.perf_counter()

        # 2) detector
        det = self.det.update(float(tick["x"]), feats)
        t2 = time.perf_counter()

        # ensure meta carries cp_prob (fallback to regime_score) so Router freeze logic works
        meta = dict(det.get("meta", {}) or {})
        meta.setdefault("cp_prob", float(det["regime_score"]))

        # 3) routing
        model_name = self.router.choose(det["regime_label"], det["regime_score"], meta)
        t3 = time.perf_counter()

        # 4) model predict+update (model may update internal state from this tick)
        y_hat, _ = self.models[model_name].predict_update(tick, feats)
        t4 = time.perf_counter()

        # 5) build intervals from *current* conformal state (no new truth yet)
        vol_hint = float(feats.get("ewm_vol") or feats.get("rv") or 0.01)
        ql, qh = self.conf.interval(
            y_hat, alpha=self.alpha_main, regime_label=det["regime_label"], scale_hint=vol_hint
        )
        multi = self.conf.interval(
            y_hat, regime_label=det["regime_label"], scale_hint=vol_hint, alphas_multi=self.alpha_list
        )
        t5 = time.perf_counter()

        # Store for legacy pointer fallback (doesn't interfere with id-based path)
        self._yhat_next = float(y_hat)
        self._regime_next = det["regime_label"]

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

    # -------- persistence (snapshot/restore) --------

    def snapshot(self) -> dict:
        """
        Return a JSON-serializable snapshot of calibration-critical state.
        If models expose snapshot(), include their state as well.
        """
        c = self.conf
        snap: dict = {
            "version": 2,
            "conformal": {
                "window": c.window,
                "decay": c.decay,
                "by_regime": c.by_regime,
                "cold_scale": c.cold_scale,
                "res_global": list(c.res_global),
                "wts_global": list(c.wts_global),
                "res_by_regime": {k: list(v) for k, v in getattr(c, "_res_by_regime", {}).items()},
                "wts_by_regime": {k: list(v) for k, v in getattr(c, "_wts_by_regime", {}).items()},
            },
            "router": {
                "regime_prev": getattr(self, "_regime_prev", None),
                "regime_next": getattr(self, "_regime_next", None),
            },
            "pending": {
                # store only the values needed to recompute residuals on restore
                "items": [
                    {"pred_id": pid, "y_hat": float(v.get("y_hat", 0.0)), "regime": v.get("regime")}
                    for pid, v in self._pending.items()
                ]
            },
        }
        # Optional: capture model state
        model_states: dict[str, dict] = {}
        for name, m in self.models.items():
            try:
                snap_fn = getattr(m, "snapshot", None)
                if callable(snap_fn):
                    model_states[name] = snap_fn()  # type: ignore[no-any-return]
            except Exception:
                pass
        if model_states:
            snap["models"] = model_states
        return snap

    def restore(self, snap: dict) -> None:
        """
        Restore state from snapshot(). Core config stays; buffers/pointers/pending are restored.
        """
        if not isinstance(snap, dict):
            return
        s_conf = snap.get("conformal", {}) or {}
        c = self.conf

        # global buffers
        try:
            c.res_global = deque(list(s_conf.get("res_global", [])), maxlen=c.window)
            c.wts_global = deque(list(s_conf.get("wts_global", [])), maxlen=c.window)
        except Exception:
            c.res_global = deque(maxlen=c.window)
            c.wts_global = deque(maxlen=c.window)

        # per-regime
        if c.by_regime:
            c._res_by_regime = {}
            c._wts_by_regime = {}
            try:
                for k, arr in (s_conf.get("res_by_regime") or {}).items():
                    c._res_by_regime[str(k)] = deque(list(arr), maxlen=c.window)
                for k, arr in (s_conf.get("wts_by_regime") or {}).items():
                    c._wts_by_regime[str(k)] = deque(list(arr), maxlen=c.window)
            except Exception:
                c._res_by_regime = {}
                c._wts_by_regime = {}

        # pointers
        r = snap.get("router", {}) or {}
        self._regime_prev = r.get("regime_prev")
        self._regime_next = r.get("regime_next")
        # pending items
        self._pending = OrderedDict()
        for it in (snap.get("pending", {}) or {}).get("items", []):
            pid = str(it.get("pred_id"))
            if not pid:
                continue
            self._pending[pid] = {"y_hat": float(it.get("y_hat", 0.0)), "regime": it.get("regime")}
        # Optional: models
        mdl = snap.get("models") or {}
        for name, state in mdl.items():
            m = self.models.get(name)
            try:
                restore_fn = getattr(m, "restore", None)
                if m and callable(restore_fn):
                    restore_fn(state)  # type: ignore[arg-type]
            except Exception:
                pass
