import pandas as pd
import pytest

from scripts.compare_restored_models import evaluation_mask, verify_training_cutoff


def test_october_has_2976_issue_times_without_adjacent_month_targets():
    times=pd.Series(pd.date_range('2025-09-30T23:30Z','2025-11-01T00:00Z',freq='15min'))
    start=pd.Timestamp('2025-10-01T00:00Z'); end=pd.Timestamp('2025-11-01T00:00Z')
    chosen=times[evaluation_mask(times,start,end)]
    assert len(chosen)==31*96
    assert chosen.min()==pd.Timestamp('2025-09-30T23:45Z')
    assert chosen.max()==pd.Timestamp('2025-10-31T23:30Z')


def test_frozen_september_candidate_is_valid_for_october_but_overlap_is_rejected():
    start=pd.Timestamp('2025-10-01T00:00Z')
    verify_training_cutoff({'train_target_end_max':'2025-09-01T00:00Z'},start)
    with pytest.raises(ValueError):
        verify_training_cutoff({'train_target_end_max':'2025-10-01T00:15Z'},start)
    with pytest.raises(ValueError):
        verify_training_cutoff({},start)
