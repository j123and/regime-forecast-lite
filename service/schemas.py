from __future__ import annotations

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    timestamp: str  # consider timezone-aware datetime
    x: float
    covariates: dict[str, float] | None = Field(default_factory=dict)

class PredictOut(BaseModel):
    y_hat: float
    interval_low: float
    interval_high: float
    regime: str
    score: float
    latency_ms: dict[str, float]
    warmup: bool = False
    degraded: bool = False
