from __future__ import annotations

from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable


class Tick(TypedDict):
    timestamp: str
    x: float
    covariates: NotRequired[dict[str, float]]  # optional


# Flexible mapping of feature name -> value
Features = dict[str, float]


class DetectorOut(TypedDict):
    regime_label: str
    regime_score: float
    meta: dict[str, object]


@runtime_checkable
class OnlineModel(Protocol):
    def predict_update(self, tick: Tick, feats: Features) -> tuple[float, dict[str, Any]]: ...
