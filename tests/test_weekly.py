from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.weekly import STEP, WeeklyExperiment, floor, qualified


class ConstantModel:
    def predict(self, rows):
        return np.full(len(rows), 6.0)


class CaptureModel(ConstantModel):
    def fit(self, x, y):
        self.x, self.y = x, y
        return self


def history(end, days=2):
    times = pd.date_range(end-timedelta(days=days), end-STEP, freq='15min')
    return pd.DataFrame({'value':5.0,'observation':times+timedelta(minutes=5)},index=times)


def artifact(now):
    return {'model':ConstantModel(),'version':'frozen-version',
            'trained_at':(now-timedelta(days=1)).isoformat(),'train_rows':1000}


def test_context_never_uses_future_values_or_outside_last_day():
    now = floor(datetime.now(UTC))
    frame = history(now)
    before = WeeklyExperiment.context(frame, now)
    frame.loc[now] = [1000, now]
    frame.loc[now-timedelta(days=2),'value'] = 1000
    after = WeeklyExperiment.context(frame, now)
    assert before == after
    assert after['mean'] == 5
    assert after['coverage'] == 1


def test_quality_requires_physical_fof2_and_known_confidence():
    assert qualified(5,50) and qualified(5,999)
    assert not qualified(5,-1)
    assert not qualified(float('nan'),90)
    assert not qualified(5,49)


def test_daily_issue_has_672_future_intervals_and_survives_restart(tmp_path):
    now = datetime.now(UTC)
    run = WeeklyExperiment(tmp_path)
    run.issue('TEST',history(floor(now)),artifact(now),now)
    points = run.db.execute('SELECT * FROM predictions ORDER BY target_start').fetchall()
    assert len(points) == 672
    assert datetime.fromisoformat(points[0]['target_start']) > now
    assert datetime.fromisoformat(points[-1]['target_end'])-datetime.fromisoformat(points[0]['target_start']) == timedelta(days=7)
    assert all(p['actual'] is None for p in points)
    run.close()
    run = WeeklyExperiment(tmp_path)
    run.issue('TEST',history(floor(now)),artifact(now),datetime.now(UTC))
    assert run.db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 672
    run.close()


def test_score_waits_for_completed_interval_and_freezes_first_fact(tmp_path):
    now = datetime.now(UTC)
    run = WeeklyExperiment(tmp_path)
    run.issue('TEST',history(floor(now)),artifact(now),now)
    point = run.db.execute('SELECT * FROM predictions ORDER BY target_start LIMIT 1').fetchone()
    start = datetime.fromisoformat(point['target_start'])
    end = datetime.fromisoformat(point['target_end'])
    facts = pd.DataFrame({'value':[5.5],'observation':[start+timedelta(minutes=5)]},index=[pd.Timestamp(start)])
    run.score('TEST',facts,start+timedelta(minutes=10))
    assert run.db.execute('SELECT COUNT(*) FROM predictions WHERE actual IS NOT NULL').fetchone()[0] == 0
    run.score('TEST',facts,end+STEP)
    facts.value = 99
    run.score('TEST',facts,end+2*STEP)
    assert run.db.execute('SELECT actual FROM predictions WHERE actual IS NOT NULL').fetchone()[0] == 5.5
    run.publish('TEST',facts,artifact(now),end+STEP)
    import json
    report = json.loads((tmp_path/'stations/TEST.json').read_text())
    metric = report['metrics'][0]
    assert metric['count'] == 1
    assert metric['due_count'] == 2
    assert metric['missing_count'] == 1
    assert metric['coverage_percent'] == 50
    run.close()


def test_stale_and_insufficient_context_does_not_issue(tmp_path):
    now = datetime.now(UTC)
    run = WeeklyExperiment(tmp_path)
    run.issue('STALE',history(floor(now)-timedelta(hours=3)),artifact(now),now)
    run.issue('SHORT',history(floor(now)).iloc[-20:],artifact(now),now)
    assert run.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
    run.close()


def test_no_retraining_before_seven_days(tmp_path):
    import joblib
    now = datetime.now(UTC)
    run = WeeklyExperiment(tmp_path)
    frozen = artifact(now)
    joblib.dump(frozen,tmp_path/'model.joblib')
    assert run.train_if_due({},now)['version'] == frozen['version']
    with pytest.raises(ValueError,match='Not enough'):
        run.train_if_due({},now+timedelta(days=7))
    assert joblib.load(tmp_path/'model.joblib')['version'] == frozen['version']
    run.close()


def test_training_excludes_targets_after_cutoff(monkeypatch, tmp_path):
    now = floor(datetime.now(UTC))
    frame = history(now,days=30)
    future = history(now+timedelta(days=4),days=4)
    future.value = 900
    frame = pd.concat([frame,future])
    model = CaptureModel()
    monkeypatch.setattr('app.services.weekly.make_pipeline',lambda *args:model)
    run = WeeklyExperiment(tmp_path)
    result = run.train_if_due({'TEST':frame},now)
    assert len(model.y) >= 1000
    assert set(model.y) == {5.0}
    assert datetime.fromisoformat(result['latest_training_target_end']) <= now
    assert max(row['lead_days'] for row in model.x) > 6
    run.close()


def test_weekly_endpoint_only_reads_snapshots_and_flags_stale(monkeypatch,tmp_path):
    import json
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv('IONOSCOPE_WEEKLY_DIR',str(tmp_path))
    client = TestClient(app)
    assert client.get('/api/v1/weekly/TEST').status_code == 503
    (tmp_path/'stations').mkdir()
    old = datetime.now(UTC)-timedelta(days=2)
    (tmp_path/'stations/TEST.json').write_text(json.dumps({
        'station':'TEST','updated_at':old.isoformat(),'generated_at':old.isoformat()}))
    response = client.get('/api/v1/weekly/test')
    assert response.status_code == 200
    assert response.json()['worker_stale'] is True
    assert response.json()['forecast_stale'] is True
