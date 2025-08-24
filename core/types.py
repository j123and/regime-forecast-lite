# core/types.py
from __future__ import annotations

from typing import TypedDict


class Tick(TypedDict):
    timestamp: str
    x: float
    covariates: dict[str, float]


class Features(TypedDict, total=False):
    # what FeatureExtractor returns
    ewm_mean: float
    ewm_var: float
    ewm_std: float
    warmup: bool
    # optional extras (harmless if absent)
    ewm_abs_dev: float
    zscore: float


class DetectorOut(TypedDict):
    cp_prob: float
    regime: str
