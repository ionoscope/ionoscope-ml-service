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
python scripts/evaluate_month.py --raw-dir ../archive/data/raw/giro --output ../experiment/evaluation --train-start 2024-09-01T00:00:00Z --test-start 2025-09-01T00:00:00Z --test-end 2025-10-01T00:00:00Z --cs-filter threshold --min-cs 50
```

The fixed model predicts the next interval mean using the previous completed
interval, causal lags and calendar features. Missing current foF2 skips the
forecast. Persistence is scored on exactly the same available pairs. The output
includes examples, predictions, metrics, coverage, a chart and a separate model;
keep these generated artifacts outside version control.

This is an archival evaluation: observation publication latency and later archive
revisions are not known. No test-period retraining or tuning is performed.

Threshold mode is the default: only physical foF2 with known finite CS >= 50
is used for inputs and labels. Unknown CS (including -1) is excluded. `--cs-filter off`
is an explicit experimental option. The script trains a separate offline model;
its metrics do not evaluate or replace the artifact currently served by the API.
Each run writes `leakage_audit.json`: removing test-month raw rows must leave
training inputs/labels unchanged, and three truncated raw replays must reproduce
earlier inputs and predictions exactly. A failed audit stops the evaluation.

To inspect the actual deployed artifact without training or replacing it, use
`scripts/audit_served_model.py --output <local-output-dir> --raw-dir <GIRO-archive>`
with the development stack running. It checks every catalog station, compares
local artifact predictions with ML API, and reports missing input features.
Its GIRO-only monthly check is explicitly diagnostic when historical external
drivers are missing. See [the served model audit](docs/served-model-audit-2026-10-24.md).


## Restoring the production scheme for a historical year/month

`scripts/restore_historical_features.py --run-dir <run-directory>` reads a frozen
`cohort.json`, GIRO raw archives under `giro/raw/giro`, and GFZ/OMNI CSVs under
`drivers/processed`. It uses the production normalization and feature builder,
preserving the 152-column schema of the existing artifact. `--resume` reuses
prepared outputs only when source, code and output hashes match.

Historical availability rules delay hourly OMNI means by one hour, Kp/ap by
three hours, and daily Ap/F10.7 by one day. These are conservative period-end
assumptions, not measured publication latencies. Missing observations remain
missing, unknown CS is rejected, and training inputs/labels must remain unchanged
when all test-month GIRO rows are physically removed.

Train a separate candidate with explicit bounds and the frozen schema:

```powershell
python scripts/train_production_model.py --features-dir <run-directory>/features --output <run-directory>/candidate/model.joblib --train-start 2024-09-01T00:00:00Z --test-start 2025-09-01T00:00:00Z --test-end 2025-10-01T00:00:00Z --feature-schema production_models/fof2_next_15min.joblib --data-manifest <run-directory>/preparation.json --model-version 0.2.0-historical-candidate
python scripts/compare_restored_models.py --run-dir <run-directory> --candidate <run-directory>/candidate/model.joblib
```

No row cap or test-period tuning is used. The candidate keeps 300 boosting
iterations and disables random early stopping; the test month is never used
for model fitting. Training saves input hashes, station codes, period boundaries,
preparation rules, per-station scores, coverage and persistence metrics. The
comparison uses identical pairs for the original artifact, candidate and baseline.
Generated datasets/models stay outside Git. A candidate expecting delayed drivers
must be deployed with matching runtime preparation rules; it is not automatically
installed into the running API.

For a later month, keep the existing candidate unchanged and pass explicit
`--history-start`, `--test-start`, `--test-end` and `--feature-schema` to the
restoration script. The comparison script also accepts `--test-start` and
`--test-end`; `--expected-candidate-sha256` enforces the frozen model identity.
It rejects candidates whose training target end overlaps the evaluation period
or whose training cutoff is unknown. Comparison never fits a model.

The October 2025 run uses `../output/october_holdout_20261024/frozen_plan.json`
to record the unchanged September candidate, original cohort, evaluation dates,
and provenance of the September observations used only as lag history.

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

## Live shadow comparison

For a PC-independent trial, see [GitHub Actions collection](docs/cloud-actions-trial.md).
Its schedule is opt-in, state is checkpointed outside Git, and the cloud journal
is kept separate from local observations. It does not host or switch the public API.

The Compose stack runs a separate worker comparing the frozen candidate, served
artifact and persistence on the same published generation. User requests keep
using the original artifact. See [runtime contract, scoring and rollback](docs/live-shadow-2026-10-25.md).
`GET /api/v1/shadow/status` reports paired errors, coverage and latency;
zero scored pairs or unavailable inputs do not establish a quality result.
