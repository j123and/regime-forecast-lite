from typing import Dict
from .types import Tick
from .features import FeatureExtractor
from .detect.bocpd import BOCPD
from .router import Router
from .conformal import OnlineConformal
from models.arima import ARIMAModel
from models.xgb import XGBModel

class Pipeline:
    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        fe_cfg = self.cfg.get('features', {})
        self.fe = FeatureExtractor(**{k: v for k, v in fe_cfg.items() if k in {'win','rv_win','ewm_alpha'}})
        dcfg = self.cfg.get('detector', {})
        self.det = BOCPD(
            threshold=dcfg.get('threshold', 0.6),
            cooldown=dcfg.get('cooldown', 5),
            hazard=dcfg.get('hazard', 1/200),
            rmax=dcfg.get('rmax', 400),
            mu0=dcfg.get('mu0', 0.0),
            kappa0=dcfg.get('kappa0', 1e-3),
            alpha0=dcfg.get('alpha0', 1.0),
            beta0=dcfg.get('beta0', 1.0),
            vol_threshold=dcfg.get('vol_threshold', 0.02),
        )
        rcfg = self.cfg.get('router', {})
        self.router = Router(dwell_min=rcfg.get('dwell_min', 10))
        self.models = {"arima": ARIMAModel(), "xgb": XGBModel()}
        self.conf = OnlineConformal()

    def process(self, tick: Tick) -> Dict[str, float | str]:
        feats = self.fe.update(tick)
        det = self.det.update(tick["x"], feats)
        model_name = self.router.choose(det["regime_label"], det["regime_score"])
        y_hat, _ = self.models[model_name].predict_update(tick, feats)
        ql, qh = self.conf.interval(y_hat, alpha=0.1)
        return {
            "y_hat": y_hat,
            "interval_low": ql,
            "interval_high": qh,
            "regime": det["regime_label"],
            "score": det["regime_score"],
        }
