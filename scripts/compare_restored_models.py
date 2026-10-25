"""Compare the original served artifact, restored candidate and persistence."""
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
sys.modules.setdefault('_loss',_loss)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_production_model import metric_record


def evaluation_mask(times, start, end):
    return (times+pd.Timedelta(minutes=15)>=start) & (times+pd.Timedelta(minutes=30)<=end)


def verify_training_cutoff(metadata, start):
    trained_until=metadata.get('train_target_end_max')
    if not trained_until or pd.to_datetime(trained_until,utc=True)>start:
        raise ValueError('Candidate training labels overlap evaluation period or lack provenance')
    return trained_until


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--original',default='production_models/fof2_next_15min.joblib')
    parser.add_argument('--test-start',default='2025-09-01T00:00:00Z')
    parser.add_argument('--test-end',default='2025-10-01T00:00:00Z')
    parser.add_argument('--expected-candidate-sha256')
    args=parser.parse_args();run=Path(args.run_dir)
    candidate_hash=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest()
    if args.expected_candidate_sha256 and candidate_hash!=args.expected_candidate_sha256:
        raise ValueError('Candidate differs from frozen evaluation plan')
    old=joblib.load(args.original);new=joblib.load(args.candidate)
    start,end=[pd.to_datetime(value,utc=True) for value in [args.test_start,args.test_end]]
    if not start < end:
        raise ValueError('Require test-start < test-end')
    trained_until=verify_training_cutoff(new['metadata'],start)
    assert old['feature_columns']==new['feature_columns']
    columns=new['feature_columns'];parts=[];stations=[]
    for path in sorted((run/'features/stations').glob('*_features.csv')):
        wanted=set(columns)|{'station','time_utc','foF2_target_15min'}
        frame=pd.read_csv(path,usecols=lambda c:c in wanted,
            dtype={c:'float32' for c in columns+['foF2_target_15min']})
        times=pd.to_datetime(frame.time_utc,utc=True)
        test=frame[evaluation_mask(times,start,end)].copy()
        eligible=np.isfinite(test.foF2)
        scored=eligible & np.isfinite(test.foF2_target_15min)
        rows=test.loc[scored].copy()
        rows['original']=old['model'].predict(rows[columns]) if len(rows) else np.nan
        rows['candidate']=new['model'].predict(rows[columns]) if len(rows) else np.nan
        rows['actual']=rows.foF2_target_15min;rows['persistence']=rows.foF2
        code=path.stem.removesuffix('_features')
        stations.append({'station':code,'slots':len(test),'forecasts':int(eligible.sum()),'scored':len(rows),
            'original':metric_record(rows.actual,rows.original),
            'candidate':metric_record(rows.actual,rows.candidate),
            'persistence':metric_record(rows.actual,rows.persistence),
            'feature_presence_fraction':float(test.loc[eligible,columns].notna().to_numpy().mean()) if eligible.any() else None})
        parts.append(rows[['station','time_utc','actual','original','candidate','persistence']])
    all_rows=pd.concat(parts,ignore_index=True)
    for name in ['original','candidate','persistence']:
        all_rows[name+'_error']=abs(all_rows[name]-all_rows.actual)
    all_rows['day']=(pd.to_datetime(all_rows.time_utc,utc=True)+pd.Timedelta(minutes=15)).dt.strftime('%Y-%m-%d')
    daily=all_rows.groupby('day').agg(n=('actual','size'),new_error=('candidate_error','sum'),
                                   base_error=('persistence_error','sum'))
    rng=np.random.default_rng(42);bootstrap=[]
    for _ in range(5000):
        sample=daily.iloc[rng.integers(0,len(daily),len(daily))]
        bootstrap.append(100*(1-sample.new_error.sum()/sample.base_error.sum()))
    assert hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest()==candidate_hash
    report={'original_sha256':hashlib.sha256(Path(args.original).read_bytes()).hexdigest(),
            'candidate_sha256':candidate_hash,
            'feature_count':len(columns),'stations':stations,'scored_pairs':len(all_rows),
            'test_start':str(start),'test_end':str(end),'candidate_train_target_end_max':trained_until,
            'unavailable_stations':[r for r in json.loads((run/'preparation.json').read_text())['stations'] if r['status']!='prepared'],
            'original':metric_record(all_rows.actual,all_rows.original),
            'candidate':metric_record(all_rows.actual,all_rows.candidate),
            'persistence':metric_record(all_rows.actual,all_rows.persistence),
            'mae_improvement_percent':100*(1-all_rows.candidate_error.sum()/all_rows.persistence_error.sum()),
            'day_cluster_bootstrap_95_percent':np.quantile(bootstrap,[.025,.975]).tolist(),
            'limits':['Fixed evaluation period; no fitting or tuning performed by comparison script',
                      'Archived definitive drivers assumed available after period end; actual publication latency unknown',
                      'Original artifact training provenance incomplete; candidate uses explicit year/month split']}
    (run/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    all_rows.to_csv(run/'comparison_predictions.csv',index=False)
    daily.to_csv(run/'daily_comparison.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_rows=[s for s in stations if s['scored']]
    y=np.arange(len(plot_rows)); fig,ax=plt.subplots(figsize=(11, max(5,len(y)*.45)))
    for offset,name,color in [(-.25,'original','#999999'),(0,'candidate','#2878b5'),(.25,'persistence','#e58b27')]:
        ax.barh(y+offset,[s[name]['mae'] for s in plot_rows],height=.23,label=name,color=color)
    ax.set_yticks(y,[s['station'] for s in plot_rows]);ax.invert_yaxis()
    ax.set(xlabel='MAE (MHz), identical scored pairs',title=f'{start:%B %Y}: production feature schema, fixed models')
    ax.legend();fig.tight_layout();fig.savefig(run/'station_mae.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='stations'},indent=2))


if __name__=='__main__':main()
