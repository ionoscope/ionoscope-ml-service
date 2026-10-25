from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

from scripts import cloud_artifacts as artifacts
from scripts.run_cloud_shadow import compact_history, models, summary


def make_state(root):
    (root / 'shadow').mkdir(parents=True)
    with sqlite3.connect(root / 'shadow/shadow.sqlite3') as db:
        db.execute('CREATE TABLE observations(value REAL)')
        db.execute('INSERT INTO observations VALUES (5.5)')
    (root / 'cloud_run.json').write_text(json.dumps({'environment': 'github-actions'}))
    (root / 'shadow/status.json').write_text('{}')


def test_checkpoint_keeps_database_and_history_but_excludes_models_logs_and_generations(tmp_path):
    state = tmp_path / 'state'
    make_state(state)
    (state / 'private-token.log').write_text('not a checkpoint member')
    (state / 'candidate.joblib').write_bytes(b'not a checkpoint member')
    history = state / 'history/raw/giro/station=TR169/history.txt'
    history.parent.mkdir(parents=True)
    history.write_text('2026-10-25T00:00:00Z 90 5.5\n')
    archive = tmp_path / 'state.zip'
    artifacts.snapshot(state, archive)
    restored = tmp_path / 'restored'
    artifacts.extract_checkpoint(archive, restored)
    assert (restored / history.relative_to(state)).read_text() == history.read_text()
    assert not (restored / 'candidate.joblib').exists()
    assert not (restored / 'private-token.log').exists()
    with sqlite3.connect(restored / 'shadow/shadow.sqlite3') as db:
        assert db.execute('SELECT value FROM observations').fetchone()[0] == 5.5


@pytest.mark.parametrize('member', ['../escape.txt', '/absolute.txt', 'history/../../escape.txt', 'history\\..\\..\\escape.txt', 'history/C:escape.txt', 'shadow/candidate.joblib'])
def test_checkpoint_rejects_unsafe_or_unexpected_paths_before_writing(tmp_path, member):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('cloud_run.json', '{}')
        bundle.writestr(member, 'bad')
    with pytest.raises(ValueError, match='Unexpected checkpoint'):
        artifacts.extract_checkpoint(archive, tmp_path / 'restored')
    assert not (tmp_path / 'restored/cloud_run.json').exists()


def test_expired_state_cannot_silently_create_a_new_run(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, 'api', lambda *a, **k: {'artifacts': [
        {'name': artifacts.PREFIX + '1', 'id': 1, 'expired': True}]})
    monkeypatch.setattr(artifacts, 'trusted', lambda a: True)
    with pytest.raises(ValueError, match='expired'):
        artifacts.restore(tmp_path, initialize=True)


def test_first_run_requires_explicit_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, 'api', lambda *a, **k: {'artifacts': []})
    with pytest.raises(ValueError, match='initialize=true'):
        artifacts.restore(tmp_path)
    assert artifacts.restore(tmp_path, initialize=True) is None


def test_restore_understands_actions_outer_zip_and_restores_latest_rerun(tmp_path, monkeypatch):
    original = tmp_path / 'original'
    make_state(original)
    inner = tmp_path / 'cloud-checkpoint.zip'
    artifacts.snapshot(original, inner)
    outer = tmp_path / 'actions.zip'
    with zipfile.ZipFile(outer, 'w') as bundle:
        bundle.write(inner, 'cloud-checkpoint.zip')
    artifact = {'name': artifacts.PREFIX + '1-1', 'id': 1, 'expired': False,
                'archive_download_url': 'https://api.github.com/example',
                'digest': 'sha256:' + hashlib.sha256(outer.read_bytes()).hexdigest()}
    monkeypatch.setattr(artifacts, 'api', lambda *a, **k: {'artifacts': [artifact]})
    monkeypatch.setattr(artifacts, 'trusted', lambda a: True)
    monkeypatch.setattr(artifacts, 'download', lambda url, path: path.write_bytes(outer.read_bytes()))
    assert artifacts.restore(tmp_path / 'restored') == 1
    assert (tmp_path / 'restored/shadow/shadow.sqlite3').exists()


def test_cross_host_redirect_never_forwards_github_credentials():
    from urllib.request import Request
    handler = artifacts.SafeRedirect()
    original = Request('https://api.github.com/start', headers={'Authorization': 'Bearer test-secret'})
    redirected = handler.redirect_request(original, None, 302, 'found', {}, 'https://example.blob.core.windows.net/signed')
    assert redirected.get_header('Authorization') is None
    with pytest.raises(ValueError, match='Insecure'):
        handler.redirect_request(original, None, 302, 'found', {}, 'http://example.com')


def test_compaction_bounds_raw_history_and_does_not_keep_future_like_invalid_tokens(tmp_path):
    import pandas as pd
    merged, destination = tmp_path / 'merged', tmp_path / 'history'
    raw = merged / 'raw/giro/station=TR169/history.txt'
    raw.parent.mkdir(parents=True)
    raw.write_text('2026-10-01T00:00:00Z 90 5\n2026-10-25T00:00:00Z 90 6\ninvalid line\n')
    (merged / 'processed').mkdir()
    pd.DataFrame({'time_utc': ['2026-10-01T00:00Z', '2026-10-25T00:00Z'], 'value': [5, 6]}).to_csv(merged / 'processed/geophysical_indices.csv', index=False)
    compact_history(merged, destination, datetime(2026, 10, 16, tzinfo=UTC))
    assert (destination / 'raw/giro/station=TR169/history.txt').read_text() == '2026-10-25T00:00:00Z 90 6\n'
    assert len(pd.read_csv(destination / 'processed/geophysical_indices.csv')) == 1


def test_model_bytes_are_verified_before_loading(tmp_path):
    (tmp_path / 'candidate.joblib').write_bytes(b'not trusted')
    plan = {'models': {'candidate': {'file': 'candidate.joblib', 'sha256': 'wrong'}}}
    with pytest.raises(ValueError, match='SHA-256'):
        models(plan, tmp_path)


def test_cleanup_never_deletes_other_workflows_artifacts(monkeypatch):
    rows = [{'name': artifacts.PREFIX + str(i), 'id': i} for i in range(5)]
    rows.append({'name': 'unrelated-build', 'id': 0})
    deleted = []
    def api(path, delete=False):
        if delete:
            deleted.append(path)
        else:
            return {'artifacts': rows}
    monkeypatch.setattr(artifacts, 'api', api)
    monkeypatch.setattr(artifacts, 'trusted', lambda row: row['id'] != 1)
    artifacts.cleanup(keep=3)
    assert deleted == ['/actions/artifacts/0']
