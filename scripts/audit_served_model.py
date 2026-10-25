"""Audit the deployed artifact without fitting or replacing any model.

Live historical rows are a retrospective feature-table check, not an arrival-time
replay. The monthly GIRO-only run reports missing drivers explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ionoscope-ml-service'))
sys.path.insert(0, str(ROOT / 'ionoscope-data'))
from app.services.model_loader import load_production_model, DEFAULT_MODEL_PATH
from feature_engineering.build_features import build_station_features, load_config
from data_preparation.normalize_time_grid import read_giro_raw, normalize_giro
from scripts.evaluate_month import metrics


def score(frame, artifact, start, end):
    frame = target_confidence(frame)
    frame['time_utc'] = pd.to_datetime(frame.time_utc, utc=True)
    frame = frame.sort_values('time_utc').set_index('time_utc')
    if frame.index.duplicated().any():
        raise ValueError('Duplicate feature timestamps')
    step = pd.Timedelta(minutes=15)
    # Exact next interval lookup; never shift across missing timestamps.
    frame['actual'] = frame.foF2.reindex(frame.index + step).to_numpy()
    frame['target_start'] = frame.index + step
    frame = frame[(frame.target_start >= start) & (frame.target_start + step <= end)].copy()
    inputs = frame.reindex(columns=artifact['feature_columns']).apply(pd.to_numeric, errors='coerce')
    current = pd.to_numeric(frame.foF2, errors='coerce')
    valid = current.between(.5, 25)
    actual = pd.to_numeric(frame.actual, errors='coerce')
    valid &= actual.between(.5, 25)
    if 'CS' in frame:
        cs = pd.to_numeric(frame.CS, errors='coerce')
        target_cs = pd.to_numeric(frame.target_cs, errors='coerce')
        valid &= np.isfinite(cs) & cs.ge(50) & np.isfinite(target_cs) & target_cs.ge(50)
    if 'foF2_source_timestamp' in frame:
        observed = pd.to_datetime(frame.foF2_source_timestamp, utc=True, errors='coerce')
        valid &= observed.notna() & (observed < frame.index + step)
    scored = frame.loc[valid].copy()
    scored['prediction'] = artifact['model'].predict(inputs.loc[valid]) if valid.any() else np.nan
    model_metrics = metrics(scored.actual, scored.prediction)
    baseline = metrics(scored.actual, scored.foF2)
    missing = [c for c in artifact['feature_columns'] if c not in frame or inputs[c].isna().all()]
    return {'slots': len(frame), 'scored_pairs': len(scored), 'model': model_metrics,
            'persistence': baseline, 'mae_improvement_percent':
                100 * (1 - model_metrics['mae_mhz'] / baseline['mae_mhz']) if baseline['mae_mhz'] else None,
            'missing_features': missing, 'feature_presence_fraction': float(inputs.notna().to_numpy().mean()) if len(inputs) else 0}, scored


def target_confidence(frame):
    frame = frame.copy()
    frame['time_utc'] = pd.to_datetime(frame.time_utc, utc=True)
    if 'CS' in frame:
        lookup = frame.set_index('time_utc').CS
        frame['target_cs'] = lookup.reindex(frame.time_utc + pd.Timedelta(minutes=15)).to_numpy()
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    artifact = load_production_model()
    if artifact is None:
        raise ValueError('No production model')
    workspace = ROOT / 'ionoscope-data/live_data'
    state = json.loads((workspace / 'state.json').read_text())
    feature_dir = workspace / state['generation'] / 'features/stations'
    registry = json.loads((workspace / 'station_registry.json').read_text())
    station_rows = {row['station']: row for row in registry['stations']}
    metadata = pd.read_csv(ROOT / 'ionoscope-data/data_preparation/configs/stations_metadata.csv')
    code_column = 'station' if 'station' in metadata else 'code'
    all_codes = set(metadata[code_column].dropna().astype(str)) | set(station_rows)
    raw = read_giro_raw(Path(args.raw_dir))
    historical_codes = set(raw.station)
    counts = Counter()
    for trees in artifact['model']._predictors:
        for tree in trees:
            for node in tree.nodes:
                if not node['is_leaf']:
                    counts[artifact['feature_columns'][int(node['feature_idx'])]] += 1
    report = {'artifact_sha256': hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
              'metadata': artifact['metadata'], 'feature_count': len(artifact['feature_columns']),
              'used_features_split_counts': dict(counts), 'live_generation': state['generation'],
              'stations': [], 'monthly_giro_only': [], 'api_parity': []}
    start, end = pd.Timestamp('2025-09-01', tz='UTC'), pd.Timestamp('2025-10-01', tz='UTC')
    with httpx.Client(timeout=30, trust_env=False) as client:
        report['api_model_info'] = client.get('http://localhost:8002/api/v1/model-info').raise_for_status().json()
        assert report['api_model_info']['feature_count'] == len(artifact['feature_columns'])
        assert report['api_model_info']['train_rows'] == artifact['metadata']['train_rows']
        for code in sorted(all_codes):
            path = feature_dir / f'{code}_features.csv'
            row = {'station': code, 'registry_active': station_rows.get(code, {}).get('active'),
                   'diagnosis': station_rows.get(code, {}).get('diagnosis'),
                   'full_month_raw_available': code in historical_codes,
                   'live_features_available': path.exists()}
            if path.exists():
                frame = target_confidence(pd.read_csv(path))
                row['data_start'] = str(frame.time_utc.min()); row['data_end'] = str(frame.time_utc.max())
                stats, scored = score(frame, artifact, frame.time_utc.min(), frame.time_utc.max() + pd.Timedelta(minutes=15))
                stats['used_but_missing_features'] = [c for c in stats['missing_features'] if c in counts]
                row['recent_table_check'] = stats
                scored.to_csv(out / f'{code}_recent_predictions.csv')
                latest = client.get(f'http://localhost:8001/api/v1/features/{code}/latest')
                if latest.status_code == 200:
                    payload = latest.json()
                    features = payload['features']
                    vector = pd.DataFrame([{c: features.get(c, np.nan) for c in artifact['feature_columns']}])
                    local = round(float(artifact['model'].predict(vector)[0]), 3)
                    response = client.post('http://localhost:8002/api/v1/predict', json={'station':code,'features':features})
                    response.raise_for_status()
                    assert response.json()['prediction'] == local
                    report['api_parity'].append({'station':code,'prediction':local,'equal':True})
            report['stations'].append(row)
    config = load_config(ROOT / 'ionoscope-data/feature_engineering/configs/feature_engineering.json')
    for code in sorted(historical_codes):
        selected = raw[raw.station == code].copy()
        selected = selected[(selected.CS >= 50) & np.isfinite(selected.CS)]
        selected.loc[~selected.foF2.between(.5,25), 'foF2'] = np.nan
        normalized = normalize_giro(selected, '15min', None, '30min')
        grid = pd.date_range(start - pd.Timedelta(days=7), end - pd.Timedelta(minutes=15), freq='15min')
        normalized = normalized.set_index('time_utc').reindex(grid).rename_axis('time_utc').reset_index()
        normalized['station'] = code
        table_path = out / f'{code}_monthly_grid.csv'; normalized.to_csv(table_path, index=False)
        frame = build_station_features(table_path, config, metadata.rename(columns={code_column:'station'}))
        frame = target_confidence(frame)
        stats, scored = score(frame, artifact, start, end)
        stats.update(station=code, scope='GIRO-only diagnostic; missing drivers; not complete production replay')
        stats['used_but_missing_features'] = [c for c in stats['missing_features'] if c in counts]
        report['monthly_giro_only'].append(stats)
        scored.to_csv(out / f'{code}_monthly_predictions.csv')
    report['limits'] = ['Recent feature tables contain archive revisions and are not full month tests',
                        'Monthly inputs lack historical external drivers',
                        'Artifact training provenance and pre-fix split cannot be reconstructed from metadata alone',
                        'No model fitting, production cache changes or artifact replacement performed']
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'stations':len(report['stations']), 'live_tables':sum(r['live_features_available'] for r in report['stations']),
                      'api_parity':len(report['api_parity']), 'monthly':report['monthly_giro_only']}, indent=2))


if __name__ == '__main__':
    main()
