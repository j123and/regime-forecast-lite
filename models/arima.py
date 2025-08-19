from __future__ import annotations

from typing import Any

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
    _HAS_SM = True
except Exception:
    SARIMAX = None  # type: ignore[assignment]
    SARIMAXResults = None  # type: ignore[assignment]
    _HAS_SM = False

def _select_feature_order(feats: dict[str, float] | None, exog_keys: list[str] | None) -> list[str]:
    if exog_keys is not None:
        return list(exog_keys)
    if not feats:
        return []
    default = ["z", "ewm_vol", "ac1", "rv"]
    present = [k for k in default if k in feats]
    return present if present else sorted(feats.keys())

class ARIMAModel:
    """
    Rolling SARIMAX with sparse refits and fast `append` between refits.
    Predicts y_{t+1|t}. Fits only on the last `window` observations.
    Falls back to naive RW if statsmodels is unavailable or fails.
    """

    def __init__(
        self,
        window: int = 500,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        refit_every: int = 50,
        exog_keys: list[str] | None = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        maxiter: int = 50,
    ) -> None:
        self.window = int(window)
        self.order = order
        self.seasonal_order = seasonal_order
        self.refit_every = int(refit_every)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = int(maxiter)

        self.exog_keys: list[str] | None = exog_keys
        self._feat_order: list[str] = []

        self._y: list[float] = []
        self._X: list[list[float]] = []
        self._res: SARIMAXResults | None = None  # type: ignore[name-defined]
        self._since_refit: int = 0

    def _vectorize(self, feats: dict[str, float]) -> list[float]:
        if not self._feat_order:
            self._feat_order = _select_feature_order(feats, self.exog_keys)
        return [float(feats[k]) for k in self._feat_order] if self._feat_order else []

    def _maybe_refit(self) -> bool:
        if not _HAS_SM:
            return False
        n = len(self._y)
        if n < max(10, sum(self.order) + 5):
            return False
        if self._res is None or self._since_refit >= self.refit_every:
            y = self._y[-self.window :]
            x_exog = self._X[-self.window :] if self._feat_order else None
            try:
                model = SARIMAX(  # type: ignore[misc]
                    y,
                    exog=x_exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility,
                )
                self._res = model.fit(disp=False, maxiter=self.maxiter)  # type: ignore[assignment]
                self._since_refit = 0
                return True
            except Exception:
                self._res = None
                return False
        return False

    def predict_update(self, tick: dict[str, Any], feats: dict[str, float]) -> tuple[float, dict]:
        x_t = float(tick["x"])
        x_vec = self._vectorize(feats)

        self._y.append(x_t)
        if self._feat_order:
            self._X.append(x_vec)
        if len(self._y) > self.window:
            self._y = self._y[-self.window :]
        if self._feat_order and len(self._X) > self.window:
            self._X = self._X[-self.window :]

        appended = False
        if _HAS_SM and self._res is not None:
            try:
                self._res = self._res.append(  # type: ignore[assignment]
                    endog=[x_t],
                    exog=[x_vec] if self._feat_order else None,
                    refit=False,
                )
                self._since_refit += 1
                appended = True
            except Exception:
                self._res = None

        refit = self._maybe_refit()

        if _HAS_SM and self._res is not None:
            try:
                x_next = [x_vec] if self._feat_order else None
                fc = self._res.get_forecast(steps=1, exog=x_next)
                y_hat = float(fc.predicted_mean[0])
                return y_hat, {
                    "model": "sarimax",
                    "refit": refit,
                    "appended": appended,
                    "nobs": len(self._y),
                    "exog_cols": self._feat_order,
                }
            except Exception:
                pass

        return x_t, {
            "model": "naive",
            "refit": False,
            "appended": appended,
            "nobs": len(self._y),
            "exog_cols": self._feat_order,
        }
