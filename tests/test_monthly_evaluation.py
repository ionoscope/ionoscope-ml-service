import numpy as np
import pandas as pd

from scripts.evaluate_month import STEP, build_examples, split_examples, prepare_observations


def example_frame():
    times = pd.date_range('2025-08-24', '2025-10-01', freq='15min', tz='UTC')
    values = np.arange(len(times), dtype=float)
    return pd.DataFrame({'time_utc': times, 'foF2': values,
                         'foF2_source_timestamp': times + pd.Timedelta(minutes=10)})


def test_interval_is_only_available_at_its_end_and_predicts_next_interval():
    start = pd.Timestamp('2025-09-01', tz='UTC')
    frame = build_examples(example_frame(), start, pd.Timestamp('2025-10-01', tz='UTC'))
    row = frame.iloc[10]
    assert row.origin == row.source_interval_start + STEP
    assert row.target_start == row.origin
    assert row.target_end == row.origin + STEP
    assert row.actual == frame.iloc[11].foF2
    assert row.foF2_lag_1 == frame.iloc[9].foF2


def test_train_labels_do_not_cross_month_boundary():
    start = pd.Timestamp('2025-08-25', tz='UTC')
    cutoff = pd.Timestamp('2025-09-01', tz='UTC')
    end = pd.Timestamp('2025-10-01', tz='UTC')
    frame = build_examples(example_frame(), start, end)
    train, test = split_examples(frame, start, cutoff, end)
    assert train.target_end.max() <= cutoff
    assert train.target_start.min() >= start
    assert test.target_start.min() == cutoff
    assert test.target_end.max() == end
    assert len(test) == 30 * 24 * 4


def test_future_observations_cannot_change_earlier_model_inputs():
    start = pd.Timestamp('2025-09-01', tz='UTC')
    end = pd.Timestamp('2025-10-01', tz='UTC')
    cutoff = pd.Timestamp('2025-09-15', tz='UTC')
    raw = example_frame()
    before = build_examples(raw, start, end)
    raw.loc[raw.time_utc >= cutoff, 'foF2'] = -999
    after = build_examples(raw, start, end)
    features = ['foF2'] + [c for c in before if c.startswith(('foF2_lag_', 'foF2_diff_', 'foF2_mean_'))]
    pd.testing.assert_frame_equal(before.loc[before.origin <= cutoff, features],
                                  after.loc[after.origin <= cutoff, features])


def test_threshold_rejects_unknown_cs_and_low_cs_for_inputs_and_labels():
    times = pd.date_range('2025-09-01', periods=6, freq='15min', tz='UTC')
    raw = pd.DataFrame({'station': 'TR169', 'time_utc': times, 'foF2': [5.] * 6,
                        'CS': [70., 20., np.nan, -1., np.inf, 50.]})
    normalized = prepare_observations(raw, 50)
    assert normalized.foF2.notna().tolist() == [True, False, False, False, False, True]
    assert prepare_observations(raw, None).foF2.notna().all()
    frame = build_examples(normalized, times[0], times[-1] + STEP)
    first = frame[frame.source_interval_start == times[0]].iloc[0]
    assert first.foF2 == 5 and pd.isna(first.actual)
