from __future__ import annotations

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    timestamp: str
    x: float
    covariates: dict[str, float] = Field(default_factory=dict)

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

class FeedbackIn(BaseModel):
    y_true: float
    timestamp: str | None = None

class FeedbackOut(BaseModel):
    regime: str
    hits: dict[str, bool]
