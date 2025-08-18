from typing import Dict, TypedDict

class Tick(TypedDict):
    timestamp: str
    x: float
    covariates: Dict[str, float]

class Features(TypedDict, total=False):
    z: float  # placeholder feature
    # add more fields as features grow

class DetectorOut(TypedDict):
    regime_label: str
    regime_score: float
    meta: Dict[str, float]
