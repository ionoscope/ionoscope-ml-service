"""Verify a frozen model's earlier inputs/predictions against truncated raw archives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn._loss import _loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ionoscope-ml-service'))
sys.path.insert(0,str(ROOT/'ionoscope-data'))
sys.modules.setdefault('_loss',_loss)
from scripts.restore_historical_features import read_drivers, station_grid
from data_preparation.normalize_time_grid import read_giro_raw
from feature_engineering.build_features import build_station_features, load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--candidate',required=True)
    args=parser.parse_args();run=Path(args.run_dir);model_path=Path(args.candidate)
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest();artifact=joblib.load(model_path)
    plan=json.loads((run/'frozen_plan.json').read_text())
    assert digest==plan['candidate_sha256']
    start,end,history=[pd.to_datetime(plan[k],utc=True) for k in ['test_start','test_end','history_start']]
    grid=pd.date_range(history,end,freq='15min',inclusive='left');cols=artifact['feature_columns']
    config=load_config(ROOT/'ionoscope-data/feature_engineering/configs/feature_engineering.json')
    metadata=pd.read_csv(ROOT/'ionoscope-data/data_preparation/configs/stations_metadata.csv')[['station','latitude','longitude_180']].drop_duplicates('station')
    staging=run/'replay';staging.mkdir(exist_ok=True);prefixes=[]
    for fraction in [.05,.5,.95]:
        boundary=(start+(end-start)*fraction).floor('15min');directory=staging/f'{boundary:%Y%m%dT%H%M}'
        directory.mkdir(exist_ok=True)
        for name in ['geophysical_indices.csv','omni_solar_wind.csv']:
            data=pd.read_csv(run/'drivers/processed'/name)
            data=data[pd.to_datetime(data.time_utc,utc=True)<boundary]
            data.to_csv(directory/name,index=False)
        prefixes.append((boundary,read_drivers(directory,grid)))
    report={'candidate_sha256':digest,'no_fitting':True,'stations':[]}
    for path in sorted((run/'features/stations').glob('*_features.csv')):
        code=path.stem.removesuffix('_features')
        full=pd.read_csv(path,usecols=lambda c:c in set(cols)|{'time_utc'},dtype={c:'float32' for c in cols})
        times=pd.to_datetime(full.time_utc,utc=True)
        raw=read_giro_raw(run/f'giro/raw/giro/station={code}',include_stations={code})
        row={'station':code,'checks':[]}
        for boundary,drivers in prefixes:
            before=station_grid(raw[raw.time_utc<boundary],grid,drivers)
            before['station']=code;norm=staging/f'{code}_prefix.csv';before.to_csv(norm,index=False)
            inputs=build_station_features(norm,config,metadata)[cols].astype('float32')
            mask=(times+pd.Timedelta(minutes=15)>=start)&(times+pd.Timedelta(minutes=15)<=boundary)
            pd.testing.assert_frame_equal(full.loc[mask,cols],inputs.loc[mask,cols],check_exact=True)
            eligible=mask & full.foF2.notna();delta=None
            if eligible.any():
                a=artifact['model'].predict(full.loc[eligible,cols]);b=artifact['model'].predict(inputs.loc[eligible,cols])
                delta=float(np.max(np.abs(a-b)));assert delta==0
            row['checks'].append({'raw_cutoff':str(boundary),'forecasts_compared':int(eligible.sum()),'max_prediction_difference':delta})
        report['stations'].append(row)
        (run/'replay_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==digest
    print('REPLAY AUDIT PASSED',flush=True)


if __name__=='__main__':main()
