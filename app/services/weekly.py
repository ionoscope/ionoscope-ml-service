"""Independent weekly experiment. Never replaces the production model.

Historical archive is used for training only. Scores are exclusively prospective:
issued intervals must start after calculation and facts are frozen on first receipt.
"""
from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

STEP = timedelta(minutes=15)
MODEL_NAME = 'weekly_hgb_last24h_v1'


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError('UTC timestamp required')
    return value.to_pydatetime().astimezone(UTC)


def floor(value):
    return value.replace(minute=value.minute//15*15, second=0, microsecond=0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.tmp') as handle:
        json.dump(value, handle, allow_nan=False)
        temporary = handle.name
    os.replace(temporary, path)


def qualified(value, cs):
    try:
        value, cs = float(value), float(cs)
        return math.isfinite(value) and .5 <= value <= 25 and (50 <= cs <= 100 or cs == 999)
    except (ValueError, TypeError):
        return False


class WeeklyExperiment:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root/'weekly.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS batches (name TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS raw (
                station TEXT, observed TEXT, value REAL, PRIMARY KEY(station, observed));
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, station TEXT, issue_day TEXT, generated TEXT,
                model_version TEXT, input_end TEXT, observation TEXT, coverage REAL,
                trained_at TEXT, train_rows INTEGER,
                UNIQUE(station, issue_day));
            CREATE TABLE IF NOT EXISTS predictions (
                run_id TEXT, target_start TEXT, target_end TEXT, prediction REAL,
                lead_hours REAL, actual REAL, received TEXT,
                PRIMARY KEY(run_id, target_start));
            CREATE INDEX IF NOT EXISTS targets ON predictions(target_start);
        ''')

    def close(self):
        self.db.close()

    def ingest(self, workspace):
        """Only completed immutable batches; retain first quality-qualified raw value."""
        done = {r[0] for r in self.db.execute('SELECT name FROM batches')}
        paths = sorted((Path(workspace)/'batches').glob('*/processed/giro_scaled.csv'))
        for path in paths:
            batch = path.parents[1]
            if batch.name in done or not (batch/'run_manifest.json').exists():
                continue
            rows = []
            with path.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    if not qualified(row.get('foF2'), row.get('CS')):
                        continue
                    try:
                        observed = utc(row['time_utc']).isoformat()
                    except (ValueError, KeyError, TypeError):
                        continue
                    rows.append((row['station'], observed, float(row['foF2'])))
            with self.db:
                self.db.executemany('INSERT OR IGNORE INTO raw VALUES(?,?,?)', rows)
                self.db.execute('INSERT INTO batches VALUES(?)', (batch.name,))

    def frames(self, now):
        result = {}
        for station, in self.db.execute('SELECT DISTINCT station FROM raw'):
            rows = self.db.execute('SELECT observed,value FROM raw WHERE station=? AND observed>=? AND observed<? ORDER BY observed',
                (station, (now-timedelta(days=120)).isoformat(), floor(now).isoformat())).fetchall()
            if not rows:
                continue
            frame = pd.DataFrame([dict(r) for r in rows])
            frame['observed'] = pd.to_datetime(frame.observed, utc=True, format='mixed')
            frame['slot'] = frame.observed.dt.floor('15min')
            result[station] = frame.groupby('slot').agg(value=('value','mean'), observation=('observed','max'))
        return result

    @staticmethod
    def context(frame, input_end):
        window = frame.loc[(frame.index >= input_end-timedelta(days=1)) & (frame.index < input_end)]
        if len(window) < 48:
            return None
        values = window.value.to_numpy()
        profile = {t.hour*4+t.minute//15: float(row.value) for t,row in window.iterrows()}
        return {'profile': profile, 'mean': float(values.mean()), 'std': float(values.std()),
                'last': float(values[-1]), 'coverage': len(window)/96,
                'input_end': input_end, 'observation': utc(window.observation.iloc[-1])}

    @staticmethod
    def features(station, context, target_start):
        slot = target_start.hour*4+target_start.minute//15
        profile = context['profile']
        return {'station': station, 'day_value': profile.get(slot, context['mean']),
                'slot_missing': float(slot not in profile),
                'mean': context['mean'], 'std': context['std'], 'last': context['last'],
                'coverage': context['coverage'],
                'lead_days': (target_start+STEP-context['input_end']).total_seconds()/86400,
                'hour_sin': math.sin(2*math.pi*slot/96), 'hour_cos': math.cos(2*math.pi*slot/96),
                'year_sin': math.sin(2*math.pi*target_start.timetuple().tm_yday/365.25),
                'year_cos': math.cos(2*math.pi*target_start.timetuple().tm_yday/365.25)}

    def train_if_due(self, frames, now):
        path = self.root/'model.joblib'
        if path.exists():
            artifact = joblib.load(path)
            if utc(artifact['trained_at']) > now:
                raise ValueError('Model training timestamp is in the future')
            if (now-utc(artifact['trained_at'])).total_seconds() < 7*86400:
                return artifact
        x, y, target_times, horizons = [], [], [], []
        for station, frame in frames.items():
            for anchor in pd.date_range(frame.index.min().ceil('D')+timedelta(days=1),
                                        now-timedelta(days=1), freq='D'):
                context = self.context(frame, anchor)
                if context is None:
                    continue
                targets = frame.loc[(frame.index >= anchor) & (frame.index < anchor+timedelta(days=7))
                                    & (frame.index+STEP <= now)]
                # Rotate the sampled quarter-hour across training days; infer every quarter-hour.
                for t,row in targets.iterrows():
                    if t.minute//15 != anchor.dayofyear % 4:
                        continue
                    x.append(self.features(station, context, t))
                    y.append(float(row.value))
                    target_times.append(t+STEP)
                    horizons.append((t+STEP-anchor).total_seconds()/3600)
        if len(x) < 1000 or sum(h >= 144 for h in horizons) < 30:
            raise ValueError('Not enough qualified history to train a seven-day model')
        model = make_pipeline(DictVectorizer(sparse=False), HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1,
            early_stopping=False, random_state=42))
        model.fit(x, y)
        artifact = {'model': model, 'trained_at': now.isoformat(), 'train_rows': len(x),
                    'latest_training_target_end': max(target_times).isoformat(),
                    'training_policy': 'All training labels end before training cutoff; no prospective scores use training predictions',
                    'model_name': MODEL_NAME, 'version': uuid4().hex}
        temporary = self.root/'model.tmp'
        joblib.dump(artifact, temporary)
        os.replace(temporary, path)
        atomic_json(self.root/'model.json', {k:v for k,v in artifact.items() if k != 'model'})
        return artifact

    def issue(self, station, frame, artifact, now):
        day = now.date().isoformat()
        if self.db.execute('SELECT 1 FROM runs WHERE station=? AND issue_day=?', (station, day)).fetchone():
            return
        if not len(frame):
            return
        input_end = utc(frame.index[-1])+STEP
        context = self.context(frame, input_end)
        if context is None or (now-context['observation']).total_seconds() > 120*60:
            return
        # Start after inference too: a slow calculation cannot turn observations into forecasts.
        starts = [floor(now)+STEP+i*STEP for i in range(672)]
        values = artifact['model'].predict([self.features(station, context, t) for t in starts])
        finished = datetime.now(UTC)
        if finished >= starts[0] or finished < now:
            return
        run_id = uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)',
                (run_id, station, day, finished.isoformat(), artifact['version'], input_end.isoformat(),
                 context['observation'].isoformat(), context['coverage'],artifact['trained_at'],artifact['train_rows']))
            self.db.executemany('INSERT INTO predictions(run_id,target_start,target_end,prediction,lead_hours) VALUES(?,?,?,?,?)',
                [(run_id,t.isoformat(),(t+STEP).isoformat(),float(np.clip(v,.5,25)),
                  (t+STEP-finished).total_seconds()/3600) for t,v in zip(starts,values,strict=True)])

    def score(self, station, frame, now):
        pending = self.db.execute('''SELECT p.run_id,p.target_start,r.generated FROM predictions p
            JOIN runs r ON p.run_id=r.id WHERE r.station=? AND p.actual IS NULL AND p.target_end<=?''',
            (station, now.isoformat())).fetchall()
        with self.db:
            for row in pending:
                target = pd.Timestamp(row['target_start'])
                if target in frame.index and utc(row['generated']) < utc(target):
                    self.db.execute('UPDATE predictions SET actual=?,received=? WHERE run_id=? AND target_start=? AND actual IS NULL',
                        (float(frame.loc[target,'value']),now.isoformat(),row['run_id'],row['target_start']))

    def publish(self, station, frame, artifact, now):
        run = self.db.execute('SELECT * FROM runs WHERE station=? ORDER BY generated DESC LIMIT 1', (station,)).fetchone()
        if run is None:
            return
        points = [dict(r) for r in self.db.execute('SELECT target_start,target_end,prediction,actual FROM predictions WHERE run_id=? ORDER BY target_start', (run['id'],))]
        all_scores = self.db.execute('''SELECT p.prediction,p.actual,p.lead_hours,p.target_end FROM predictions p
            JOIN runs r ON r.id=p.run_id WHERE r.station=? AND r.generated>=?''',
            (station,(now-timedelta(days=30)).isoformat())).fetchall()
        metrics = []
        for label, lo, hi in [('0–24 ч',0,24),('24–72 ч',24,72),('72–168 ч',72,169)]:
            group = [r for r in all_scores if lo <= r['lead_hours'] < hi]
            due = sum(utc(r['target_end']) <= now for r in group)
            errors = [r['prediction']-r['actual'] for r in group if r['actual'] is not None]
            metrics.append({'horizon':label,'count':len(errors),'issued_count':len(group),
                'due_count':due,'missing_count':due-len(errors),
                'coverage_percent':round(len(errors)/due*100,2) if due else None,
                'mae': float(np.mean(np.abs(errors))) if errors else None,
                'rmse': float(np.sqrt(np.mean(np.square(errors)))) if errors else None})
        observed = frame.loc[frame.index >= floor(now)-timedelta(days=1)]
        atomic_json(self.root/'stations'/f'{station}.json', {
            'station':station,'model':MODEL_NAME,'model_version':run['model_version'],
            'generated_at':run['generated'],'updated_at':now.isoformat(),
            'trained_at':run['trained_at'],'next_training_at':(utc(artifact['trained_at'])+timedelta(days=7)).isoformat(),
            'training_rows':run['train_rows'],'input_end':run['input_end'],
            'observation_timestamp':run['observation'],'input_coverage_percent':round(run['coverage']*100,2),
            'step_minutes':15,'points':points,'metrics':metrics,'automatic_promotion':False,
            'observed':[{'t':(t+STEP).isoformat(),'value':float(r.value)} for t,r in observed.iterrows()],
            'evaluation_note':'Prospective only; horizon groups differ from the current 15-minute model. No automatic promotion.'})

    def cycle(self, workspace):
        self.ingest(workspace)
        now = datetime.now(UTC)
        frames = self.frames(now)
        artifact = self.train_if_due(frames, now)
        # Fitting may take time. Inputs and labels were frozen at the earlier cutoff.
        now = datetime.now(UTC)
        state = json.loads((Path(workspace)/'state.json').read_text(encoding='utf-8'))
        stations = state.get('available_stations', state.get('stations', []))
        for station in stations:
            if station not in frames:
                continue
            self.score(station, frames[station], now)
            self.issue(station, frames[station], artifact, datetime.now(UTC))
            self.publish(station, frames[station], artifact, datetime.now(UTC))
        return {'phase':'ok','updated_at':datetime.now(UTC).isoformat(),
                'trained_at':artifact['trained_at'],'stations':stations,'automatic_promotion':False}
