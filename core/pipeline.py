from typing import Dict
from .types import Tick
from .features import FeatureExtractor
from .detect.bocpd import BOCPD
from .router import Router
from .conformal import OnlineConformal
from models.arima import ARIMAModel
from models.xgb import XGBModel

class Pipeline:
    def __init__(self) -> None:
        self.fe = FeatureExtractor()
        self.det = BOCPD()
        self.router = Router()
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
