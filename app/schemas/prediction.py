from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionRequest(BaseModel):
    station: str
    features: dict[str, float] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    station: str
    parameter: str
    prediction: float
    unit: str
    horizon: str
    model: str
    model_version: str


class ModelInfoResponse(BaseModel):
    artifact_loaded: bool
    parameter: str
    horizon: str
    unit: str
    model: str
    model_version: str
    feature_count: int
    train_rows: int | None = None
    test_rows: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
