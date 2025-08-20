from __future__ import annotations

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    timestamp: str
    x: float
    covariates: dict[str, float] = Field(default_factory=dict)
    # New: identify the stream and (optionally) the intended target timestamp.
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
    # New: keys for idempotency & OOO handling
    prediction_id: str
    series_id: str
    target_timestamp: str


class TruthIn(BaseModel):
    # Must provide either prediction_id OR (series_id AND target_timestamp).
    prediction_id: str | None = None
    series_id: str | None = None
    target_timestamp: str | None = None
    # Accept any of these for the value
    y: float | None = None
    y_true: float | None = None
    value: float | None = None


class TruthOut(BaseModel):
    status: str
    matched_by: str | None = None  # "prediction_id" | "series_ts" | None
    idempotent: bool | None = None
    queued: bool | None = None
