from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.schemas.prediction import ModelInfoResponse, PredictionRequest, PredictionResponse
from app.services.model_loader import load_production_model


DEFAULT_FOF2_MHZ = 5.0
DEFAULT_HORIZON = "15min"
DEFAULT_MODEL_VERSION = "0.1.0"
DEFAULT_PARAMETER = "foF2"
DEFAULT_UNIT = "MHz"


def predict_fof2_next(request: PredictionRequest) -> PredictionResponse:
    artifact = load_production_model()
    if artifact is not None:
        return predict_with_artifact(request, artifact)

    prediction = request.features.get("foF2")
    if prediction is None:
        prediction = request.features.get("foF2_state", DEFAULT_FOF2_MHZ)

    return PredictionResponse(
        station=request.station.upper(),
        parameter=DEFAULT_PARAMETER,
        prediction=round(float(prediction), 3),
        unit=DEFAULT_UNIT,
        horizon=DEFAULT_HORIZON,
        model="fallback_persistence",
        model_version=DEFAULT_MODEL_VERSION,
    )


def predict_with_artifact(request: PredictionRequest, artifact: dict) -> PredictionResponse:
    feature_columns = list(artifact["feature_columns"])
    model = artifact["model"]
    metadata = dict(artifact.get("metadata", {}))
    row = {feature: request.features.get(feature, np.nan) for feature in feature_columns}
    frame = pd.DataFrame([row], columns=feature_columns)
    prediction = float(model.predict(frame)[0])

    return PredictionResponse(
        station=request.station.upper(),
        parameter=str(metadata.get("parameter", metadata.get("target", DEFAULT_PARAMETER))),
        prediction=round(prediction, 3),
        unit=str(metadata.get("unit", DEFAULT_UNIT)),
        horizon=str(metadata.get("horizon", DEFAULT_HORIZON)),
        model=str(metadata.get("model", type(model).__name__)),
        model_version=str(metadata.get("model_version", DEFAULT_MODEL_VERSION)),
    )


def get_model_info() -> ModelInfoResponse:
    artifact = load_production_model()
    if artifact is None:
        return ModelInfoResponse(
            artifact_loaded=False,
            parameter=DEFAULT_PARAMETER,
            horizon=DEFAULT_HORIZON,
            unit=DEFAULT_UNIT,
            model="fallback_persistence",
            model_version=DEFAULT_MODEL_VERSION,
            feature_count=0,
            metrics={},
        )

    metadata = dict(artifact.get("metadata", {}))
    metrics = metadata.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    return ModelInfoResponse(
        artifact_loaded=True,
        parameter=str(metadata.get("parameter", metadata.get("target", DEFAULT_PARAMETER))),
        horizon=str(metadata.get("horizon", DEFAULT_HORIZON)),
        unit=str(metadata.get("unit", DEFAULT_UNIT)),
        model=str(metadata.get("model", type(artifact["model"]).__name__)),
        model_version=str(metadata.get("model_version", DEFAULT_MODEL_VERSION)),
        feature_count=len(artifact["feature_columns"]),
        train_rows=metadata.get("train_rows"),
        test_rows=metadata.get("test_rows"),
        metrics=normalize_metrics(metrics),
    )


def normalize_metrics(metrics: dict) -> dict[str, float]:
    clean_metrics = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean_metrics[str(key)] = number
    return clean_metrics
