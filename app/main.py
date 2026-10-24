from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas.prediction import (
    HealthResponse,
    ErrorResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from .services.prediction_service import get_model_info, predict_fof2_next


app = FastAPI(
    title="IonoScope ML Service",
    version="0.1.0",
    description="Minimal foF2 next-step inference service for IonoScope.",
    responses={503: {"model": ErrorResponse}},
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ionoscope-ml-service")


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_fof2_next(request)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "Model could not produce a prediction"}) from exc


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    try:
        return get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "model_unavailable", "message": "Model artifact could not be loaded"}) from exc
