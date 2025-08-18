from typing import Optional, Dict
from pydantic import BaseModel, Field

class PredictIn(BaseModel):
    timestamp: str
    x: float
    covariates: Optional[Dict[str, float]] = Field(default_factory=dict)

class PredictOut(BaseModel):
    y_hat: float
    interval_low: float
    interval_high: float
    regime: str
    score: float
    latency_ms: Dict[str, float]
