from __future__ import annotations

from collections import deque
from typing import Any

try:
    import xgboost as xgb  # type: ignore[import-not-found]
    _HAS_XGB = True
except Exception:
    xgb = None  # type: ignore[assignment]
    _HAS_XGB = False

try:
    from sklearn.linear_model import SGDRegressor  # type: ignore[import-not-found]
    _HAS_SK = True
except Exception:
    SGDRegressor = None  # type: ignore[assignment]
    _HAS_SK = False

def _order(features: dict[str, float], keys: list[str] | None) -> list[str]:
    if keys is not None:
        return list(keys)
    if not features:
        return []
    default = ["z", "ewm_vol", "ac1", "rv"]
    present = [k for k in default if k in features]
    return present if present else sorted(features.keys())

class XGBModel:
    """
    Sliding-window booster learning y_{t+1} from features at time t.
    No leakage: (features_t) paired with y_{t+1} when it arrives.
    Retrains every `retrain_every` ticks on last `window` pairs.
    Falls back to SGDRegressor or naive RW if libs are missing.
    """

    def __init__(
        self,
        window: int = 5000,
        retrain_every: int = 50,
        min_train: int = 200,
        feature_keys: list[str] | None = None,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        n_jobs: int = 0,
    ) -> None:
        self.window = int(window)
        self.retrain_every = int(retrain_every)
        self.min_train = int(min_train)
        self.feature_keys = feature_keys

        self._feat_order: list[str] = []
        self._X: deque[list[float]] = deque(maxlen=self.window)
        self._y: deque[float] = deque(maxlen=self.window)
        self._last_feats: list[float] | None = None
        self._ticks_since_retrain: int = 0

        self._booster: Any | None = None  # xgb.XGBRegressor or SGDRegressor
        self._is_xgb = False

        self._xgb_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="reg:squarederror",
        )

    def _vec(self, feats: dict[str, float]) -> list[float]:
        if not self._feat_order:
            self._feat_order = _order(feats, self.feature_keys)
        return [float(feats[k]) for k in self._feat_order] if self._feat_order else []

    def _fit_if_needed(self) -> bool:
        n = len(self._y)
        if n < self.min_train:
            return False
        if self._booster is None or self._ticks_since_retrain >= self.retrain_every:
            X = list(self._X)
            y = list(self._y)
            try:
                if _HAS_XGB:
                    model = xgb.XGBRegressor(**self._xgb_kwargs)  # type: ignore[operator]
                    model.fit(X, y, verbose=False)
                    self._booster = model
                    self._is_xgb = True
                elif _HAS_SK:
                    model = SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)  # type: ignore[name-defined]
                    model.fit(X, y)
                    self._booster = model
                    self._is_xgb = False
                else:
                    self._booster = None
                    self._is_xgb = False
                    return False
                self._ticks_since_retrain = 0
                return True
            except Exception:
                self._booster = None
                self._is_xgb = False
                return False
        return False

    def predict_update(self, tick: dict[str, Any], feats: dict[str, float]) -> tuple[float, dict]:
        x_t = float(tick["x"])
        f_t = self._vec(feats)

        if self._last_feats is not None:
            self._X.append(self._last_feats)
            self._y.append(x_t)

        self._ticks_since_retrain += 1
        refit = self._fit_if_needed()

        if self._booster is not None:
            try:
                y_hat = float(self._booster.predict([f_t])[0])  # type: ignore[call-arg]
            except Exception:
                y_hat = x_t
        else:
            y_hat = x_t

        self._last_feats = f_t

        meta = {
            "model": "xgb" if self._is_xgb else ("sgd" if self._booster is not None else "naive"),
            "refit": refit,
            "n_train": len(self._y),
            "feat_dim": len(self._feat_order),
            "feat_cols": self._feat_order,
        }
        return y_hat, meta
