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
import json
import os
from datetime import UTC, datetime
from pathlib import Path


app = FastAPI(
    title="IonoScope ML Service",
    version="0.1.0",
    description="Minimal foF2 next-step inference service for IonoScope.",
    responses={503: {"model": ErrorResponse}},
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ionoscope-ml-service")


@app.get('/api/v1/shadow/status', tags=['model'])
def shadow_status() -> dict:
    directory = os.getenv('IONOSCOPE_SHADOW_DIR')
    if not directory:
        return {'phase': 'disabled', 'automatic_promotion': False}
    try:
        report = json.loads((Path(directory) / 'status.json').read_text(encoding='utf-8'))
        updated = datetime.fromisoformat(report['updated_at'])
        if (datetime.now(UTC) - updated).total_seconds() > 120:
            report['phase'] = 'stale'
        return report
    except FileNotFoundError:
        return {'phase': 'starting', 'automatic_promotion': False}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail={'code': 'shadow_status_unavailable',
                                                    'message': 'Shadow report is unavailable'}) from exc


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
