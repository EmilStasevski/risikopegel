from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from risikopegel.splits import (
    LookAheadError,
    Split,
    assert_no_lookahead,
    out_of_time_split,
    purged_kfold,
    walk_forward,
)

pytestmark = pytest.mark.leakage


def _daily(n: int) -> npt.NDArray[np.datetime64]:
    out: npt.NDArray[np.datetime64] = np.datetime64("2020-01-01") + np.arange(n).astype("timedelta64[D]")
    return out


def test_walk_forward_expanding_is_strictly_ordered() -> None:
    t = _daily(100)
    folds = list(walk_forward(t, test_size=10, min_train_size=50))
    assert len(folds) == 5
    for f in folds:
        assert_no_lookahead(t, f)
        assert t[f.train_idx].min() == t[0]
    assert [f.train_idx.size for f in folds] == [50, 60, 70, 80, 90]


def test_walk_forward_rolling_window_and_gap() -> None:
    t = _daily(100)
    folds = list(walk_forward(t, test_size=5, train_size=30, gap=3))
    for f in folds:
        assert f.train_idx.size == 30
        # gap: 3 unique days strictly between train end and test start
        assert (f.test_start - f.train_end) == np.timedelta64(4, "D")
        assert_no_lookahead(t, f)


def test_walk_forward_panel_rows_with_same_date_stay_together() -> None:
    dates = np.repeat(_daily(20), 7)  # 7 assets per day
    for f in walk_forward(dates, test_size=2, min_train_size=10):
        assert not set(dates[f.train_idx]) & set(dates[f.test_idx])
        assert f.test_idx.size == 14


def test_walk_forward_unsorted_input_uses_time_not_row_order() -> None:
    rng = np.random.default_rng(1)
    t = rng.permutation(_daily(60))
    for f in walk_forward(t, test_size=10, min_train_size=20):
        assert_no_lookahead(t, f)


def test_walk_forward_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="test_size"):
        list(walk_forward(_daily(10), test_size=0))
    with pytest.raises(ValueError, match="missing"):
        list(walk_forward(np.array([1.0, np.nan]), test_size=1))


def test_out_of_time_split_with_embargo() -> None:
    t = _daily(365)
    s = out_of_time_split(t, np.datetime64("2020-06-30"), embargo=np.timedelta64(30, "D"))
    assert t[s.train_idx].max() == np.datetime64("2020-06-30")
    assert t[s.test_idx].min() == np.datetime64("2020-07-31")
    assert_no_lookahead(t, s)


def test_purged_kfold_removes_overlapping_labels() -> None:
    t = np.arange(100)
    label_end = t + 5  # each label looks 5 steps ahead
    for f in purged_kfold(t, label_end, n_splits=5, embargo=2):
        lo, hi = f.test_start, f.test_end
        tr_t, tr_end = t[f.train_idx], label_end[f.train_idx]
        assert not ((tr_t <= hi) & (tr_end >= lo)).any(), "label window overlaps test block"
        assert not ((tr_t > hi) & (tr_t <= hi + 2)).any(), "embargo violated"


def test_assert_no_lookahead_catches_overlap_and_label_windows() -> None:
    t = np.arange(10)
    bad = Split(np.arange(0, 6), np.arange(5, 10), 0, 5, 5, 9)
    with pytest.raises(LookAheadError):
        assert_no_lookahead(t, bad)
    ok = Split(np.arange(0, 5), np.arange(5, 10), 0, 4, 5, 9)
    assert_no_lookahead(t, ok)
    # label of the last train row ends inside the test period -> leak
    with pytest.raises(LookAheadError, match="training information"):
        assert_no_lookahead(t, ok, label_end_times=t + 2)


def test_random_split_is_detected() -> None:
    t = _daily(200)
    rng = np.random.default_rng(0)
    idx = rng.permutation(200)
    random_split = Split(np.sort(idx[:150]), np.sort(idx[150:]), None, None, None, None)
    with pytest.raises(LookAheadError):
        assert_no_lookahead(t, random_split)
