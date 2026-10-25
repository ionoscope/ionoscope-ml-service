from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.services.shadow import ShadowRun, PROFILE, digest


class ConstantModel:
    def predict(self, frame):
        return [5.25] * len(frame)


@pytest.fixture
def fixture(tmp_path):
    artifact = {'model': ConstantModel(), 'feature_columns': ['foF2'], 'metadata': {}}
    old, candidate = tmp_path / 'old.joblib', tmp_path / 'candidate.joblib'
    joblib.dump(artifact, old)
    joblib.dump(artifact, candidate)
    start = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)
    run = ShadowRun(tmp_path / 'run', old, candidate, digest(candidate), ['TR169', 'MISSING'], now=start)
    workspace = tmp_path / 'live'
    directory = workspace / 'generations/one/features'
    (directory / 'stations').mkdir(parents=True)
    (directory / 'input_contract.json').write_text(json.dumps({'profile': PROFILE, 'implementation_sha256': 'test-contract'}))
    state = {'generation': 'generations/one', 'input_profile': PROFILE,
             'prepared_at': (start + timedelta(seconds=1)).isoformat(), 'available_stations': ['TR169']}
    (workspace / 'state.json').write_text(json.dumps(state))
    path = directory / 'stations/TR169_features.csv'
    pd.DataFrame([{'time_utc': start - timedelta(minutes=15), 'foF2': 5,
                   'foF2_source_timestamp': start - timedelta(minutes=5)}]).to_csv(path, index=False)
    yield run, workspace, path, start, old, candidate
    run.close()


def test_prospective_pair_is_immutable_scored_only_after_target_end(fixture):
    run, workspace, path, start, *_ = fixture
    now = start + timedelta(seconds=5)
    run.cycle(workspace, now=now, clock=lambda: now)
    initial = dict(run.db.execute("SELECT * FROM slots WHERE station='TR169'").fetchone())
    assert initial['original'] == initial['candidate'] == 5.25
    assert initial['actual'] is None
    frame = pd.read_csv(path)
    frame.loc[0, 'foF2'] = 20  # revision must not regenerate the prediction
    frame.loc[1] = [start.isoformat(), 6, (start + timedelta(minutes=5)).isoformat()]
    frame.to_csv(path, index=False)
    run.cycle(workspace, now=now + timedelta(minutes=5), clock=lambda: now + timedelta(minutes=5))
    assert run.db.execute("SELECT actual FROM slots WHERE station='TR169'").fetchone()[0] is None
    later = start + timedelta(minutes=16)
    result = run.cycle(workspace, now=later, clock=lambda: later)
    row = dict(run.db.execute("SELECT * FROM slots WHERE station='TR169' AND issue=?", (start.isoformat(),)).fetchone())
    assert row['features'] == initial['features']
    assert row['actual'] == 6
    assert result['common_scored'] == 1
    assert result['candidate']['mae'] == .75
    frame.loc[1, 'foF2'] = 7
    frame.to_csv(path, index=False)
    run.cycle(workspace, now=later, clock=lambda: later)
    assert run.db.execute("SELECT actual FROM slots WHERE station='TR169' AND issue=?", (start.isoformat(),)).fetchone()[0] == 6


def test_outage_counts_slots_without_retrospective_predictions(fixture):
    run, workspace, _, start, *_ = fixture
    now = start + timedelta(hours=1, seconds=1)
    report = run.cycle(workspace, now=now, clock=lambda: now)
    assert report['opportunities'] == 10
    assert report['candidate_forecasts'] == 0
    assert report['common_scored'] == 0


def test_inference_finishing_after_target_is_excluded(fixture):
    run, workspace, _, start, *_ = fixture
    report = run.cycle(workspace, now=start + timedelta(seconds=5), clock=lambda: start + timedelta(minutes=15))
    assert report['candidate_forecasts'] == 0
    assert report['statuses']['deadline_missed'] == 1


def test_profile_mismatch_never_scores_or_predicts(fixture):
    run, workspace, _, start, *_ = fixture
    state = json.loads((workspace / 'state.json').read_text())
    state['input_profile'] = 'legacy'
    (workspace / 'state.json').write_text(json.dumps(state))
    report = run.cycle(workspace, now=start + timedelta(seconds=5))
    assert report['candidate_forecasts'] == 0
    assert report['error']


def test_restart_requires_same_models_and_cohort(fixture):
    run, _, _, start, old, candidate = fixture
    restarted = ShadowRun(run.directory, old, candidate, digest(candidate), ['TR169', 'MISSING'], now=start)
    assert restarted.manifest == run.manifest
    restarted.close()
    with pytest.raises(ValueError, match='identity changed'):
        ShadowRun(run.directory, old, candidate, digest(candidate), ['TR169'], now=start)
    with pytest.raises(ValueError, match='SHA-256'):
        ShadowRun(run.directory, old, candidate, 'wrong', ['TR169'], now=start)


def test_candidate_failure_does_not_discard_original_or_enter_common_score(fixture, monkeypatch):
    import app.services.shadow as shadow
    run, workspace, _, start, *_ = fixture
    run.candidate = dict(run.candidate)
    predict = shadow.predict_with_artifact
    def fail_candidate(request, artifact):
        if artifact is run.candidate:
            raise RuntimeError('candidate failed')
        return predict(request, artifact)
    monkeypatch.setattr(shadow, 'predict_with_artifact', fail_candidate)
    now = start + timedelta(seconds=5)
    report = run.cycle(workspace, now=now, clock=lambda: now)
    assert report['original_forecasts'] == 1
    assert report['candidate_forecasts'] == 0
    assert report['common_scored'] == 0
