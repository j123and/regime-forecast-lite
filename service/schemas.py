from __future__ import annotations

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    timestamp: str
    x: float
    covariates: dict[str, float] = Field(default_factory=dict)
    series_id: str = "default"
    target_timestamp: str | None = None  # if omitted, server will echo `timestamp`


class PredictOut(BaseModel):
    y_hat: float
    interval_low: float
    interval_high: float
    intervals: dict[str, list[float]] | None = None
    regime: str
    score: float
    latency_ms: dict[str, float]
    warmup: bool = False
    degraded: bool = False
    prediction_id: str
    series_id: str
    target_timestamp: str


class TruthIn(BaseModel):
    # NOTE: The service currently requires prediction_id (idempotent).
    prediction_id: str | None = None
    series_id: str | None = None  # reserved for future use
    target_timestamp: str | None = None  # reserved for future use
    y: float | None = None
    y_true: float | None = None
    value: float | None = None


class TruthOut(BaseModel):
    status: str
    matched_by: str | None = None  # "prediction_id" | None
    idempotent: bool | None = None
    queued: bool | None = None
