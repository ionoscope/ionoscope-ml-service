import pandas as pd
import pytest

from scripts.train_production_model import split_train_test


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
