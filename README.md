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
matrix. The split uses target interval times: training targets must finish by
`--test-start`, and test targets must start at or after it. `--dataset-step`
(default 15min) specifies the aggregation interval. A pre-boundary feature row
whose target lies in the test period is never used for training.


## Historical month evaluation

Check out `ionoscope-data` alongside this repository; the evaluation script uses
its raw GIRO reader and normalization functions. Install optional evaluation
packages with `python -m pip install -r requirements-evaluation.txt`.

```powershell
python scripts/evaluate_month.py --raw-dir ../archive/data/raw/giro --output ../experiment/evaluation --train-start 2024-09-01T00:00:00Z --test-start 2025-09-01T00:00:00Z --test-end 2025-10-01T00:00:00Z --cs-filter off
```

The fixed model predicts the next interval mean using the previous completed
interval, causal lags and calendar features. Missing current foF2 skips the
forecast. Persistence is scored on exactly the same available pairs. The output
includes examples, predictions, metrics, coverage, a chart and a separate model;
keep these generated artifacts outside version control.

This is an archival evaluation: observation publication latency and later archive
revisions are not known. No test-period retraining or tuning is performed.


## Prediction request validation

Prediction requests require a valid station code and finite numeric features,
including current `foF2` or `foF2_state`. Empty state, NaN/Infinity and invalid
station codes return 422; a synthetic default 5 MHz is not produced for empty
requests. Station codes are trimmed and normalized to uppercase.
Unreadable model artifacts or inference failures return 503 with
`detail.code = "model_unavailable"`. The existing missing-artifact persistence
fallback remains explicit in the model name. `/health` is process liveness;
`/api/v1/model-info` should be checked to confirm an actual model is loaded.


## Artifact portability

ML dependency versions are pinned to the environment of the stored model.
The loader registers the legacy `_loss` module alias used by Windows sklearn
1.8 Cython pickles, enabling this trusted artifact to load on Linux without
rewriting it. Real predictions have been checked across Windows and the Linux
Docker image. Upgrade dependencies together with artifact revalidation/retraining.
