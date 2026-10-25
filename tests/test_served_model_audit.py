import numpy as np
import pandas as pd

from scripts.audit_served_model import score


class CurrentValue:
    def predict(self, frame):
        return frame.foF2.to_numpy()


def frame_at(minutes, values, confidence):
    times = pd.Timestamp('2025-09-01', tz='UTC') + pd.to_timedelta(minutes, unit='min')
    return pd.DataFrame({'time_utc': times, 'foF2': values, 'CS': confidence,
                         'foF2_source_timestamp': times + pd.Timedelta(minutes=5)})


def test_scoring_never_shifts_actual_across_missing_interval():
    frame = frame_at([0, 30, 45], [4, 5, 6], [70, 70, 70])
    artifact = {'model': CurrentValue(), 'feature_columns': ['foF2']}
    report, scored = score(frame, artifact, pd.Timestamp('2025-09-01', tz='UTC'),
                           pd.Timestamp('2025-09-02', tz='UTC'))
    assert report['scored_pairs'] == 1
    assert scored.actual.tolist() == [6]
    assert report['model']['mae_mhz'] == 1


def test_scoring_checks_target_confidence_and_preserves_final_target_lookup():
    frame = frame_at([0, 15, 30, 45], [4, 5, 6, 7], [70, np.nan, 70, 70])
    artifact = {'model': CurrentValue(), 'feature_columns': ['foF2']}
    report, scored = score(frame, artifact, pd.Timestamp('2025-09-01', tz='UTC'),
                           pd.Timestamp('2025-09-01T01:00Z'))
    assert report['scored_pairs'] == 1
    assert scored.actual.tolist() == [7]


def test_scoring_rejects_measurement_not_yet_available_at_origin():
    frame = frame_at([0, 15], [4, 5], [70, 70])
    frame.loc[0, 'foF2_source_timestamp'] = pd.Timestamp('2025-09-01T00:15Z')
    artifact = {'model': CurrentValue(), 'feature_columns': ['foF2']}
    report, _ = score(frame, artifact, pd.Timestamp('2025-09-01', tz='UTC'),
                       pd.Timestamp('2025-09-02', tz='UTC'))
    assert report['scored_pairs'] == 0
