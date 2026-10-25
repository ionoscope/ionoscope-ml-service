"""Append-only prospective comparison. Never selects the model served by the API."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from app.schemas.prediction import PredictionRequest
from app.services.model_loader import load_production_model
from app.services.prediction_service import predict_with_artifact

PROFILE = 'candidate-cs50-period-end-v1'
STEP = timedelta(minutes=15)


def utc(value):
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00')) if not isinstance(value, datetime) else value
    if value.tzinfo is None:
        raise ValueError('Timezone required')
    return value.astimezone(UTC)


def floor_slot(value):
    return value.replace(minute=value.minute // 15 * 15, second=0, microsecond=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ShadowRun:
    def __init__(self, directory, original, candidate, expected_hash, stations, *, now=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not expected_hash or digest(candidate) != expected_hash:
            raise ValueError('Candidate must match frozen holdout SHA-256')
        self.original = load_production_model(str(original))
        self.candidate = load_production_model(str(candidate))
        if self.original is None or self.candidate is None:
            raise ValueError('Both artifacts are required; persistence fallback is forbidden')
        if self.original['feature_columns'] != self.candidate['feature_columns']:
            raise ValueError('Model schemas differ')
        self.columns = self.candidate['feature_columns']
        self.stations = sorted(set(stations))
        self.identity = {'original_sha256': digest(original), 'candidate_sha256': expected_hash,
                         'stations': self.stations, 'input_profile': PROFILE}
        self.db = sqlite3.connect(self.directory / 'shadow.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS slots (
                station TEXT NOT NULL, issue TEXT NOT NULL, target_end TEXT NOT NULL,
                status TEXT NOT NULL, generation TEXT, generated_at TEXT, features TEXT,
                original REAL, candidate REAL, persistence REAL,
                original_ms REAL, candidate_ms REAL, input_delay_seconds REAL,
                actual REAL, actual_received_at TEXT, actual_generation TEXT,
                PRIMARY KEY(station, issue));
        ''')
        previous = self.db.execute('SELECT manifest FROM run WHERE id=1').fetchone()
        if previous:
            self.manifest = json.loads(previous['manifest'])
            if any(self.manifest.get(k) != v for k, v in self.identity.items()):
                self.db.close()
                raise ValueError('Run identity changed; use a new output directory')
        else:
            self.manifest = {**self.identity, 'started_at': utc(now or datetime.now(UTC)).isoformat(),
                             'fact_policy': 'First qualified completed interval seen; revisions do not overwrite',
                             'promotion': 'Manual review only; no automatic switching'}
            with self.db:
                self.db.execute('INSERT INTO run VALUES (1, ?)', (json.dumps(self.manifest),))
        self.start = utc(self.manifest['started_at'])

    def close(self):
        self.db.close()

    def cycle(self, workspace, *, now=None, clock=None):
        clock = clock or (lambda: datetime.now(UTC))
        now = utc(now or clock())
        issue = floor_slot(now)
        first_issue = floor_slot(self.start) + (STEP if self.start != floor_slot(self.start) else timedelta())
        # Outages remain in the denominator; never retrospectively predict missed slots.
        with self.db:
            last = self.db.execute('SELECT MAX(issue) FROM slots').fetchone()[0]
            begin = max(first_issue, utc(last) + STEP) if last else first_issue
            while begin <= issue:
                self.db.executemany('INSERT OR IGNORE INTO slots(station,issue,target_end,status) VALUES(?,?,?,?)',
                    [(s, begin.isoformat(), (begin + STEP).isoformat(), 'missed_slot') for s in self.stations])
                begin += STEP
        workspace = Path(workspace)
        try:
            state = json.loads((workspace / 'state.json').read_text(encoding='utf-8'))
            generation = workspace / state['generation']
            contract = json.loads((generation / 'features/input_contract.json').read_text())
            if state.get('input_profile') != PROFILE or contract['profile'] != PROFILE:
                raise ValueError('Input contract differs from holdout')
            implementation = contract['implementation_sha256']
            if not implementation or (self.manifest.get('input_implementation_sha256') and
                    self.manifest['input_implementation_sha256'] != implementation):
                raise ValueError('Input implementation changed; a new run is required')
            if not self.manifest.get('input_implementation_sha256'):
                self.manifest['input_implementation_sha256'] = implementation
                with self.db:
                    self.db.execute('UPDATE run SET manifest=? WHERE id=1', (json.dumps(self.manifest),))
            prepared = utc(state['prepared_at'])
            if prepared > now:
                raise ValueError('Generation prepared in the future')
        except (OSError, KeyError, ValueError) as exc:
            self._status(issue, 'input_unavailable')
            return self.report(now, error=str(exc))
        for station in self.stations:
            try:
                path = generation / f'features/stations/{station}_features.csv'
                frame = pd.read_csv(path)
                if set(self.columns) - set(frame):
                    raise ValueError('Missing schema columns')
                times = pd.to_datetime(frame.time_utc, utc=True, format='mixed')
                # Freeze the first qualified fact when its interval has finished.
                pending = self.db.execute('SELECT issue FROM slots WHERE station=? AND actual IS NULL AND target_end<=?',
                                          (station, now.isoformat())).fetchall()
                for pending_row in pending:
                    target = utc(pending_row['issue'])
                    rows = frame.loc[times == target]
                    if len(rows) == 1 and self._qualified(rows.iloc[0], target):
                        with self.db:
                            self.db.execute('UPDATE slots SET actual=?,actual_received_at=?,actual_generation=? '
                                            'WHERE station=? AND issue=? AND actual IS NULL',
                                (float(rows.iloc[0].foF2), now.isoformat(), state['generation'], station, target.isoformat()))
                stored = self.db.execute('SELECT * FROM slots WHERE station=? AND issue=?', (station, issue.isoformat())).fetchone()
                if stored is None or stored['generated_at'] is not None:
                    continue
                rows = frame.loc[times == issue - STEP]
                if len(rows) != 1 or not self._qualified(rows.iloc[0], issue - STEP):
                    self._status(issue, 'current_fof2_unavailable', station)
                    continue
                if station not in state.get('available_stations', state.get('stations', [])):
                    self._status(issue, 'station_unavailable', station)
                    continue
                row = rows.iloc[0]
                # Preserve the exact finite input vector. Targets never enter inference.
                features = {c: float(row[c]) for c in self.columns if pd.notna(row[c]) and np.isfinite(float(row[c]))}
                request = PredictionRequest(station=station, features=features)
                values, durations, errors = {}, {}, []
                for name, artifact in [('original', self.original), ('candidate', self.candidate)]:
                    started = time.perf_counter()
                    try:
                        values[name] = predict_with_artifact(request, artifact).prediction
                    except Exception as exc:
                        values[name] = None
                        errors.append(name + ':' + type(exc).__name__)
                    durations[name] = 1000 * (time.perf_counter() - started)
                finished = utc(clock())
                if finished >= issue + STEP or finished < issue:
                    self._status(issue, 'deadline_missed', station)
                    continue
                with self.db:
                    self.db.execute('UPDATE slots SET status=?,generation=?,generated_at=?,features=?,original=?,candidate=?, '
                        'persistence=?,original_ms=?,candidate_ms=?,input_delay_seconds=? '
                        'WHERE station=? AND issue=? AND generated_at IS NULL',
                        ('forecast' if not errors else ';'.join(errors), state['generation'], finished.isoformat(),
                         json.dumps(features, sort_keys=True, allow_nan=False), values['original'], values['candidate'],
                         float(row.foF2), durations['original'], durations['candidate'],
                         (prepared - issue).total_seconds(), station, issue.isoformat()))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._status(issue, 'station_input_error:' + type(exc).__name__, station)
        return self.report(now)

    @staticmethod
    def _qualified(row, start):
        try:
            return (.5 <= float(row.foF2) <= 25 and
                    start <= utc(row.foF2_source_timestamp) < start + STEP)
        except (ValueError, TypeError, AttributeError):
            return False

    def _status(self, issue, status, station=None):
        with self.db:
            self.db.execute('UPDATE slots SET status=? WHERE issue=? AND generated_at IS NULL' +
                            (' AND station=?' if station else ''),
                            (status, issue.isoformat(), station) if station else (status, issue.isoformat()))

    def report(self, now, *, error=None):
        rows = [dict(r) for r in self.db.execute('SELECT station,issue,target_end,status,generated_at,original,candidate,'
            'persistence,original_ms,candidate_ms,input_delay_seconds,actual,actual_received_at '
            'FROM slots ORDER BY issue,station')]
        scored = [r for r in rows if all(r[c] is not None for c in ['original', 'candidate', 'persistence', 'actual'])]

        def metrics(selected):
            result = {'opportunities': len(selected), 'original_forecasts': sum(r['original'] is not None for r in selected),
                      'candidate_forecasts': sum(r['candidate'] is not None for r in selected)}
            common = [r for r in selected if all(r[c] is not None for c in ['original', 'candidate', 'persistence', 'actual'])]
            result['common_scored'] = len(common)
            result['coverage'] = {name: result[name + '_forecasts'] / len(selected) if selected else None
                                  for name in ['original', 'candidate']}
            for name in ['original', 'candidate', 'persistence']:
                errors = [r[name] - r['actual'] for r in common]
                result[name] = {'mae': float(np.mean(np.abs(errors))) if errors else None,
                                'rmse': float(np.sqrt(np.mean(np.square(errors)))) if errors else None}
            for baseline in ['original', 'persistence']:
                mae = result[baseline]['mae']
                result['improvement_vs_' + baseline + '_percent'] = (
                    100 * (1 - result['candidate']['mae'] / mae) if mae else None)
            return result

        result = {'manifest': self.manifest, 'updated_at': now.isoformat(), 'error': error,
                  'phase': 'input_unavailable' if error else 'collecting',
                  **metrics(rows), 'stations': {s: metrics([r for r in rows if r['station'] == s]) for s in self.stations},
                  'pending_facts': sum(r['actual'] is None and r['generated_at'] is not None for r in rows),
                  'common_days': len({r['issue'][:10] for r in scored}),
                  'statuses': {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}},
                  'days': {day: metrics([r for r in rows if r['issue'][:10] == day])
                           for day in sorted({r['issue'][:10] for r in rows})},
                  'latency': {}, 'automatic_promotion': False}
        for field in ['original_ms', 'candidate_ms', 'input_delay_seconds']:
            values = [r[field] for r in rows if r[field] is not None]
            result['latency'][field] = {'p50': float(np.quantile(values, .5)) if values else None,
                                      'p95': float(np.quantile(values, .95)) if values else None}
        for field, values in {
            'forecast_delay_seconds': [(utc(r['generated_at']) - utc(r['issue'])).total_seconds()
                                       for r in rows if r['generated_at']],
            'fact_delay_seconds': [(utc(r['actual_received_at']) - utc(r['target_end'])).total_seconds()
                                   for r in rows if r['actual_received_at']],
        }.items():
            result['latency'][field] = {'p50': float(np.quantile(values, .5)) if values else None,
                                      'p95': float(np.quantile(values, .95)) if values else None}
        out = self.directory / 'status.json'
        temporary = out.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(out)
        return result
