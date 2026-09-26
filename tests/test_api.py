from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services import prediction_service


client = TestClient(app)


class FakeModel:
    pass


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ionoscope-ml-service"}


def test_predict_falls_back_to_current_fof2_feature(monkeypatch) -> None:
    monkeypatch.setattr(prediction_service, "load_production_model", lambda: None)

    response = client.post(
        "/api/v1/predict",
        json={"station": "tr169", "features": {"foF2": 5.84, "gfz_Kp": 3.0}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "station": "TR169",
        "parameter": "foF2",
        "prediction": 5.84,
        "unit": "MHz",
        "horizon": "15min",
        "model": "fallback_persistence",
        "model_version": "0.1.0",
    }


def test_model_info_reports_fallback_when_artifact_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(prediction_service, "load_production_model", lambda: None)

    response = client.get("/api/v1/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "artifact_loaded": False,
        "parameter": "foF2",
        "horizon": "15min",
        "unit": "MHz",
        "model": "fallback_persistence",
        "model_version": "0.1.0",
        "feature_count": 0,
        "train_rows": None,
        "test_rows": None,
        "metrics": {},
    }


def test_model_info_reports_loaded_artifact(monkeypatch) -> None:
    monkeypatch.setattr(
        prediction_service,
        "load_production_model",
        lambda: {
            "model": FakeModel(),
            "feature_columns": ["foF2", "gfz_Kp"],
            "metadata": {
                "target": "foF2",
                "horizon": "15min",
                "model": "FakeModel",
                "model_version": "0.1.0",
                "train_rows": 100,
                "test_rows": 20,
                "metrics": {"mae": 0.31, "bad": "not-a-number"},
            },
        },
    )

    response = client.get("/api/v1/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "artifact_loaded": True,
        "parameter": "foF2",
        "horizon": "15min",
        "unit": "MHz",
        "model": "FakeModel",
        "model_version": "0.1.0",
        "feature_count": 2,
        "train_rows": 100,
        "test_rows": 20,
        "metrics": {"mae": 0.31},
    }
