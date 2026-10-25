# Trial: prospective model comparison in GitHub Actions

This workflow runs collection and inference on a GitHub-hosted Linux machine.
The user's PC and Docker are not needed. It does **not** host the public API,
train models, serve a frontend or replace the production model.

## Explicit activation

The scheduled job is disabled unless the repository variable
`IONOSCOPE_CLOUD_SHADOW_ENABLED` equals `true`. Manual runs remain available.
Pushing a workflow does not by itself activate recurring collection.

Before first use, publish a release `shadow-models-v1` in this ML repository
with exactly two trusted assets, outside Git history:

| Asset | SHA-256 |
|---|---|
| `original.joblib` | `c32e2157a7d066ff352c61db44ec48d33888b76fd270eb1520ef4cb708625b82` |
| `candidate.joblib` | `3f3008c698bee8b4b02a299cecda79441939c5db891db2b4f2c1183c24ee6350` |

These are the existing evaluated models, about 2.6 MB combined. Do not publish
training predictions, raw datasets, credentials or the local SQLite journal as
release assets. This repository is public, so the two release assets will be public.
Every downloaded model is SHA-256 verified **before** unpickling. No PAT or
additional repository secret is required for the public source checkouts/assets.
The short-lived workflow token is used only for checkpoint download/cleanup.

Start **Ionoscope cloud shadow trial** from Actions with `initialize=true`.
Confirm the job reaches its summary and uploads a checkpoint, then enable the
repository variable. Subsequent manual runs use `initialize=false`. To stop
recurring work, set the variable to `false` or disable the workflow in GitHub.
No GitHub changes, release upload or schedule activation are performed by the
local preparation scripts. Publishing requires the user's push approval.

## Execution and state

- A schedule requests a run at minutes 6, 21, 36 and 51 of each hour, avoiding
  the busiest minute zero. GitHub does not guarantee punctual execution.
- Each job is bounded to 12 minutes, collection/inference to 8 minutes.
  No background loop is left running between jobs. Runs are serialized;
  delayed/omitted slots remain missing and cannot be forecast retrospectively.
- Source checkout and all third-party Actions are pinned by commit SHA.
  The fixed 17-station cohort, two model identities and data commit are in
  `configs/cloud-shadow.json`. No selection by fresh errors is performed.
- The runner shares `candidate_inputs.prepare_live` and `ShadowRun` with the
  local evaluation. Its fresh journal is separate from the Windows journal,
  so platform, implementation fingerprints and starts are not mixed.
- GIRO, GFZ and OMNI are collected; NOAA is omitted because it is not part of
  the frozen feature schema. Unavailable drivers remain missing. CS >= 50,
  physical bounds, completed intervals, driver delays and forecast deadlines
  are identical to the evaluated input profile.
- Eight days are requested at initialization. Later runs use a three-hour
  overlap; canonical source history is trimmed to nine days. Retries preserve
  missing intervals and never backfill predictions using already known facts.
- Collection source failures are listed in `collection.json`. A source error
  is not interpreted as a quality improvement. The first run may have no
  forecasts because no full opportunity has occurred since journal creation.
- Facts, common-pair errors, coverage, statuses and delays are computed by the
  existing immutable journal. Human-readable metrics appear in each job summary.
  A small sample is not grounds for promotion.

The state lives in immutable Actions artifacts named
`ionoscope-cloud-state-<run-id>-<attempt>`. Each contains `cloud-checkpoint.zip`:
the cumulative SQLite journal, latest JSON reports, bounded canonical source
history and GIRO pacing state. Model binaries, generated feature CSVs, collection
batches, environment files and logs are excluded from checkpoints and Git.

The previous checkpoint is downloaded through the authenticated GitHub API.
Only this workflow on `main`, triggered manually or by the schedule in the same
repository, is trusted. Downloads and extraction are size bounded; unsafe paths,
symlinks, duplicate members, a damaged database or a changed model/input plan
stop the run. Cross-host redirects strip the GitHub token.

SQLite backup makes a consistent checkpoint even after a collection failure.
The new artifact is uploaded before cleanup; only the three newest checkpoints
from this workflow are retained. Other workflows' artifacts are not deleted.
Checkpoints are capped at 64 MiB each and retained for up to 30 days. This bounds
normal checkpoint storage to approximately 192 MiB; cleanup failures can temporarily
increase it and must be investigated. This is not a permanent backup service.
Download a checkpoint before a long pause. If the latest state expires or is
corrupt, the workflow fails rather than silently discarding the experiment.

## Local validation and review

```powershell
python -m pytest -q
python scripts/cloud_artifacts.py snapshot --state <test-state> --archive <test-output.zip>
```

The local integration replay uses recorded October inputs and simulated time
with the actual frozen models to verify collection/preparation/inference,
later scoring and checkpoint restoration. This is software validation, not
new prospective evidence; its output is under `../output/cloud_actions_trial_20261025/`.
The cloud run's observation period begins only after actual activation.
Do not reuse the local November 9 review deadline as 14 days of cloud evidence
if activation occurs later. Update the existing review follow-up to reference
the remote checkpoint and actual cloud start after the first successful cloud run.

To analyze results, download the newest artifact, extract its inner ZIP and
read `shadow/status.json` and `shadow/shadow.sqlite3`. Compare fixed model errors
on common real pairs, station/day coverage and publication/scheduling delays.
GitHub Actions is an experimental scheduled collector, not a replacement for
continuous API hosting or a timing SLA. No automatic promotion is implemented.

References: [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits),
[scheduled events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[artifacts](https://docs.github.com/en/rest/actions/artifacts).
