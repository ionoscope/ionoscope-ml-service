from __future__ import annotations

from fastapi import FastAPI

from .schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from .services.prediction_service import get_model_info, predict_fof2_next


app = FastAPI(
    title="IonoScope ML Service",
    version="0.1.0",
    description="Minimal foF2 next-step inference service for IonoScope.",
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ionoscope-ml-service")


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    return predict_fof2_next(request)


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    return get_model_info()
