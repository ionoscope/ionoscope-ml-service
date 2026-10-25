"""Restore the production feature schema with explicit archive availability rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ionoscope-data'))
from data_preparation.normalize_time_grid import read_giro_raw, normalize_giro, propagate_interval_features
from feature_engineering.build_features import load_config, build_station_features

STEP = pd.Timedelta(minutes=15)


from data_preparation.candidate_inputs import driver_rules, read_drivers, station_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stations', help='Optional comma-separated subset for preparation checks')
    parser.add_argument('--resume', action='store_true', help='Reuse verified outputs when inputs and preparation code match')
    parser.add_argument('--history-start', default='2024-08-24T00:00:00Z')
    parser.add_argument('--test-start', default='2025-09-01T00:00:00Z')
    parser.add_argument('--test-end', default='2025-10-01T00:00:00Z')
    parser.add_argument('--feature-schema', default=str(ROOT/'ionoscope-ml-service/production_models/fof2_next_15min.joblib'))
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    cohort = json.loads((run/'cohort.json').read_text())['stations']
    if args.stations:
        cohort = [code for code in cohort if code in args.stations.split(',')]
    history_start,cutoff,end=[pd.to_datetime(value,utc=True) for value in [args.history_start,args.test_start,args.test_end]]
    if not history_start < cutoff < end:
        raise ValueError('Require history-start < test-start < test-end')
    grid = pd.date_range(history_start, end, inclusive='left', freq='15min')
    drivers = read_drivers(run/'drivers/processed', grid)
    config = load_config(ROOT/'ionoscope-data/feature_engineering/configs/feature_engineering.json')
    metadata = pd.read_csv(ROOT/'ionoscope-data/data_preparation/configs/stations_metadata.csv')[['station','latitude','longitude_180']].drop_duplicates('station')
    from sklearn._loss import _loss
    import joblib
    sys.modules.setdefault('_loss',_loss)
    schema = joblib.load(args.feature_schema)['feature_columns']
    out = run/'features/stations'; out.mkdir(parents=True,exist_ok=True)
    norm = run/'normalized'; norm.mkdir(exist_ok=True)
    report = {'stations':[], 'driver_rules':driver_rules(), 'schema':schema,
              'history_start':str(history_start),'test_start':str(cutoff),'test_end':str(end),
              'limits':['Period-end availability assumed; publication latency unknown',
                        'Definitive/revised archives do not reproduce original arrival-time snapshots'],
              'raw_sha256':{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (run/'drivers/raw').rglob('*') if p.is_file()}}
    prior=json.loads((run/'preparation.json').read_text()) if (run/'preparation.json').exists() else {}
    cached={r['station']:r for r in prior.get('stations',[])}
    processing_files=[Path(__file__),ROOT/'ionoscope-data/data_preparation/candidate_inputs.py',ROOT/'ionoscope-data/data_preparation/normalize_time_grid.py',
                      ROOT/'ionoscope-data/feature_engineering/build_features.py',
                      ROOT/'ionoscope-data/feature_engineering/configs/feature_engineering.json',
                      ROOT/'ionoscope-data/data_preparation/configs/stations_metadata.csv']
    processing=hashlib.sha256(b''.join(p.read_bytes() for p in processing_files)+
                              json.dumps(report['raw_sha256'],sort_keys=True).encode()+
                              str((history_start,cutoff,end,schema)).encode()).hexdigest()
    for code in cohort:
        directory = run/f'giro/raw/giro/station={code}'
        source_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*.txt')}
        fingerprint=hashlib.sha256((processing+json.dumps(source_hashes,sort_keys=True)).encode()).hexdigest()
        old=cached.get(code,{})
        feature_path=out/f'{code}_features.csv'
        if args.resume and old.get('fingerprint')==fingerprint and feature_path.exists() and old.get('features_sha256')==hashlib.sha256(feature_path.read_bytes()).hexdigest():
            report['stations'].append(old);report['raw_sha256'].update(source_hashes)
            print(json.dumps({'station':code,'status':'reused_verified_preparation'}),flush=True)
            continue
        raw = read_giro_raw(directory,include_stations={code})
        entry = {'station':code,'raw_rows':len(raw)}
        if raw.empty:
            entry['status']='no_archive';report['stations'].append(entry);continue
        report['raw_sha256'].update(source_hashes)
        frame = station_grid(raw, grid, drivers)
        if frame is None:
            entry['status']='no_qualified_observations';report['stations'].append(entry);continue
        frame['station']=code
        path=norm/f'{code}_time_grid.csv';frame.to_csv(path,index=False)
        features=build_station_features(path,config,metadata)
        for c in schema:
            if c not in features:
                raise ValueError(f'{code}: missing required schema column {c}')
        # Remove test-month raw records, then recompute the training prefix.
        prefix=station_grid(raw[raw.time_utc < cutoff],grid,drivers)
        prefix['station']=code; prefix.to_csv(norm/f'{code}_prefix.csv',index=False)
        previous=build_station_features(norm/f'{code}_prefix.csv',config,metadata)
        trainmask=features.time_utc + 2*STEP <= cutoff
        pd.testing.assert_frame_equal(features.loc[trainmask,schema+['foF2_target_15min']],previous.loc[trainmask,schema+['foF2_target_15min']])
        features.to_csv(out/f'{code}_features.csv',index=False)
        present=features.loc[features.foF2.notna(),schema].notna().mean()
        entry.update(status='prepared',training_unchanged_without_test_giro=True,
                     missing_features=present.index[present.eq(0)].tolist(),
                     feature_presence_fraction=float(present.mean()),fingerprint=fingerprint,
                     features_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest())
        report['stations'].append(entry)
        print(json.dumps(entry),flush=True)
        snapshot=dict(report); snapshot['stations']+= [r for c,r in cached.items() if c not in {e['station'] for e in report['stations']}]
        (run/'preparation.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    report['stations'] += [r for c,r in cached.items() if c not in {e['station'] for e in report['stations']}]
    (run/'preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
