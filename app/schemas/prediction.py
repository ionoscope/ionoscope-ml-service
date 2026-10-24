from __future__ import annotations

from pydantic import BaseModel, Field, FiniteFloat, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class PredictionRequest(BaseModel):
    station: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9_-]+$")
    features: dict[str, FiniteFloat] = Field(default_factory=dict)

    @field_validator("station", mode="before")
    @classmethod
    def normalize_station(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_current_state(self):
        if "foF2" not in self.features and "foF2_state" not in self.features:
            raise ValueError("Current foF2 or foF2_state is required")
        return self


class PredictionResponse(BaseModel):
    station: str
    parameter: str
    prediction: FiniteFloat
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
