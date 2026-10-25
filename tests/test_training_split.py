import pandas as pd
import pytest

from scripts.train_production_model import split_train_test
from scripts.train_production_model import load_training_frame, metric_record
from scripts.train_production_model import numeric_feature_columns


@pytest.mark.parametrize('horizon', ['15min', '1h', '24h'])
def test_training_targets_never_cross_test_boundary(horizon):
    cutoff = pd.Timestamp('2025-09-01T00:00:00Z')
    frame = pd.DataFrame({'time_utc': pd.date_range(cutoff - pd.Timedelta(days=2),
                                                  cutoff + pd.Timedelta(days=1), freq='15min')})
    train, test = split_train_test(frame, str(cutoff), horizon)
    delta = pd.Timedelta(horizon)
    assert (train.time_utc + delta + pd.Timedelta(minutes=15) <= cutoff).all()
    assert (test.time_utc + delta >= cutoff).all()
    assert set(train.index).isdisjoint(test.index)


def test_last_feature_row_of_august_is_test_not_training_for_next_interval():
    frame = pd.DataFrame({'time_utc': ['2025-08-31T23:30:00Z', '2025-08-31T23:45:00Z', '2025-09-01T00:00:00Z']})
    train, test = split_train_test(frame, '2025-09-01T00:00:00Z')
    assert list(train.index) == [0]
    assert list(test.index) == [1, 2]


def test_invalid_horizon_rejected():
    frame = pd.DataFrame({'time_utc': ['2025-08-31T23:30:00Z']})
    with pytest.raises(ValueError):
        split_train_test(frame, '2025-09-01T00:00:00Z', '10min')


def test_loading_preserves_unscored_slots_and_uses_frozen_schema(tmp_path):
    path=tmp_path/'TR169_features.csv'
    path.write_text('station,time_utc,foF2,foF2_target_15min,unwanted\n'
                    'TR169,2025-09-01T00:00Z,5,,999\n'
                    'TR169,2025-09-01T00:15Z,6,7,999\n')
    frame=load_training_frame([path],'foF2_target_15min',0,0,42,['foF2'])
    assert len(frame)==2 and pd.isna(frame.iloc[0].foF2_target_15min)
    assert 'unwanted' not in frame and str(frame.foF2.dtype)=='float32'


def test_missing_station_labels_produce_json_safe_empty_metrics():
    import json
    result=metric_record(pd.Series(dtype=float),pd.Series(dtype=float))
    assert result['n']==0 and result['mae'] is None
    json.dumps(result,allow_nan=False)


def test_evaluation_outputs_and_other_targets_cannot_become_model_inputs():
    frame=pd.DataFrame({c:[1.] for c in ['foF2','CS','actual','prediction','persistence',
        'foF2_target_15min','TEC_target_1h','foF2_savgol','foF2_pred']})
    assert numeric_feature_columns(frame,'foF2')==['foF2','CS']
