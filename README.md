# IonoScope ML Service

Minimal model inference service for IonoScope.

Version `0.1.0` supports the first product scenario:

- the user chooses a station;
- `ionoscope-data` provides the latest prepared features;
- this service returns a next-step `foF2` forecast from one current model.

The service predicts `foF2` for the next 15-minute grid step. If a production
artifact is available in `production_models/fof2_next_15min.joblib`, it is used.
Otherwise the service falls back to a simple persistence prediction from the
latest `foF2` feature.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn app.main:app --reload --port 8002
```

Open API docs:

```text
http://127.0.0.1:8002/docs
```

## Endpoints

- `GET /health`
- `GET /api/v1/model-info`
- `POST /api/v1/predict`

Model info response:

```json
{
  "artifact_loaded": true,
  "parameter": "foF2",
  "horizon": "15min",
  "unit": "MHz",
  "model": "HistGradientBoostingRegressor",
  "model_version": "0.1.0",
  "feature_count": 42,
  "train_rows": 100000,
  "test_rows": 20000,
  "metrics": {
    "mae": 0.35,
    "rmse": 0.7,
    "r2": 0.96
  }
}
```

Prediction request:

```json
{
  "station": "TR169",
  "features": {
    "foF2": 5.84,
    "gfz_Kp": 3.0
  }
}
```

Prediction response:

```json
{
  "station": "TR169",
  "parameter": "foF2",
  "prediction": 5.84,
  "unit": "MHz",
  "horizon": "15min",
  "model": "fallback_persistence",
  "model_version": "0.1.0"
}
```

This service does not train models during user requests. Training is run as a
separate offline job, the exported artifact is stored under `production_models/`,
and `/api/v1/predict` only performs fast inference from prepared features.
LLM explanations should be added later in a separate orchestration or LLM
service, using this service's prediction and model metadata.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Train Production Artifact

Use prepared feature tables from the shared data pipeline:

```powershell
python .\scripts\train_production_model.py --features-dir .\features_15min --horizon 15min --output .\production_models\fof2_next_15min.joblib
```

The training script excludes future-derived target columns from the feature
matrix and evaluates on rows after `--test-start`.
