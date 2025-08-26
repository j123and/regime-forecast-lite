# core/detect/bocpd.py
from __future__ import annotations

from typing import Any

from core.config import load_config
from core.features import FeatureExtractor
from core.types import DetectorOut, Features


class BOCPD:
    """
    API compatibility with legacy ctor:
      BOCPD(threshold, hazard, rmax, mu0, kappa0, alpha0, beta0, vol_threshold=...)
    Only 'threshold' and 'vol_threshold' influence behavior here; others are accepted & ignored.

    We compute a z-like standardized surprise from features and map to cp_prob in [0,1].
    """

    def __init__(
        self,
        threshold: float | None = None,   # cp probability threshold used by tests; we map it to sensitivity
        hazard: float | None = None,      # accepted, ignored
        rmax: int | None = None,          # accepted, ignored
        mu0: float | None = None,         # accepted, ignored
        kappa0: float | None = None,      # accepted, ignored
        alpha0: float | None = None,      # accepted, ignored
        beta0: float | None = None,       # accepted, ignored
        vol_threshold: float | None = None,  # std threshold for regime split
        *,
        alpha: float | None = None,       # EWMA alpha for internal features
        min_warmup: int | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.fx = FeatureExtractor(
            alpha=float(alpha if alpha is not None else self.cfg.get("ewma_alpha", 0.1)),
            min_warmup=int(min_warmup if min_warmup is not None else self.cfg.get("min_warmup", 20)),
        )
        # Sensitivity: smaller z_threshold => more sensitive (higher cp_prob)
        if threshold is not None and threshold > 0:
            self.z_threshold = max(1e-6, 1.0 / float(threshold))  # e.g., 0.2 -> 5.0
        else:
            self.z_threshold = 3.0
        self.vol_th = float(
            vol_threshold if vol_threshold is not None else self.cfg.get("regime_vol_threshold", 0.01)
        )
        self.run_length = 0  # exposed for tests

    def _z_from_feat(self, x: float, feat: Features) -> float:
        mean = float(feat.get("ewm_mean", 0.0))
        std = float(feat.get("ewm_std", 0.0))
        if std <= 0.0:
            return 0.0
        return (x - mean) / std

    def _cp_from_z(self, z: float, warmup: bool) -> float:
        a = abs(float(z))
        if warmup:
            a *= 0.5
        p = a / self.z_threshold
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        return p

    def update(self, x: float | dict[str, Any], features: Features | None = None) -> dict[str, Any]:
        """
        Update with a new observation. Accepts raw x or a tick dict (we'll take its 'x').
        If 'features' are provided, use them; otherwise compute features internally.
        """
        if isinstance(x, dict):
            x_val = float(x.get("x", 0.0))
        else:
            x_val = float(x)

        feat: Features = features if features is not None else self.fx.update(x_val)
        z = self._z_from_feat(x_val, feat)
        cp_prob = self._cp_from_z(z, bool(feat.get("warmup", False)))

        if abs(z) >= self.z_threshold:
            self.run_length = 0
        else:
            self.run_length += 1

        regime = "volatile" if float(feat.get("ewm_std", 0.0)) >= self.vol_th else "calm"

        # Return both legacy "meta.cp_prob" and a top-level copy
        return {
            "meta": {"cp_prob": float(cp_prob)},
            "cp_prob": float(cp_prob),
            "regime": regime,
        }

    # alias some test harnesses use
    def step(self, x: float | dict[str, Any], features: Features | None = None) -> DetectorOut:
        return self.update(x, features)
