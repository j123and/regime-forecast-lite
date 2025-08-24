# service/schemas.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    timestamp: str
    x: float
    covariates: dict[str, float] | None = None
    series_id: str | None = None
    # Optional: if client already knows target ts (e.g., bar close)
    target_timestamp: str | None = None


class PredictOut(BaseModel):
    prediction_id: str
    series_id: str
    target_timestamp: str

    y_hat: float
    interval_low: float
    interval_high: float
    # Optional: map of label -> [low, high]
    intervals: dict[str, Any] | None = None

    regime: str
    score: float = 0.0  # e.g., cp probability or risk score

    latency_ms: dict[str, float] = Field(default_factory=dict)

    warmup: bool = False
    degraded: bool = False


class TruthIn(BaseModel):
    # Either provide prediction_id, or (series_id + target_timestamp)
    prediction_id: str | None = None
    series_id: str | None = None
    target_timestamp: str | None = None

    # Any of these is accepted
    y: float | None = None
    y_true: float | None = None
    value: float | None = None


class TruthOut(BaseModel):
    status: str  # "ok"
    matched_by: str  # "prediction_id" | "series+timestamp"
    idempotent: bool = False
