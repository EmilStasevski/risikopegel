"""Time-based split utilities. Random splits on temporal data are not offered on purpose.

Split policy (docs/split_policy.md):
- Final evaluation: walk-forward (expanding or rolling window), test strictly after train, optional gap.
- Hyperparameter tuning: only inside the training window of a walk-forward fold, via purged + embargoed
  K-fold (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7) or a nested walk-forward.
- Credit (application data): out-of-time split by application date, optional embargo of one
  performance window.

All functions work on *unique sorted time points*, so panel data (many rows per date) is split by date,
never by row: rows sharing a timestamp always land on the same side.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]


class LookAheadError(AssertionError):
    """Raised when a split or feature uses information from the future."""


@dataclass(frozen=True)
class Split:
    """Row indices of one fold plus the time bounds that define it (inclusive)."""

    train_idx: IntArray
    test_idx: IntArray
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any


def _as_array(times: npt.ArrayLike) -> npt.NDArray[Any]:
    arr = np.asarray(times)
    if arr.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if arr.size == 0:
        raise ValueError("times is empty")
    if arr.dtype.kind == "O":
        raise TypeError("times must be numeric or datetime64, not object dtype")
    if (arr.dtype.kind == "M" and np.isnat(arr).any()) or (arr.dtype.kind == "f" and np.isnan(arr).any()):
        raise ValueError("times contains missing values")
    return arr


def walk_forward(
    times: npt.ArrayLike,
    *,
    test_size: int,
    train_size: int | None = None,
    min_train_size: int = 1,
    step: int | None = None,
    gap: int = 0,
) -> Iterator[Split]:
    """Rolling-origin splits over unique time points.

    Sizes are counted in unique time points (e.g. days or months), not rows.

    test_size: periods per test block. train_size: rolling window length; None = expanding window.
    step: how far the origin advances per fold (default test_size, i.e. non-overlapping test blocks).
    gap: periods dropped between train end and test start (embargo, e.g. forecast horizon - 1 or the
    length of a label window, so a label at the end of train cannot see test-period outcomes).
    """
    arr = _as_array(times)
    if test_size < 1 or min_train_size < 1 or gap < 0:
        raise ValueError("test_size and min_train_size must be >= 1, gap >= 0")
    if train_size is not None and train_size < 1:
        raise ValueError("train_size must be >= 1 or None")
    step = test_size if step is None else step
    if step < 1:
        raise ValueError("step must be >= 1")

    uniq, inverse = np.unique(arr, return_inverse=True)
    n = uniq.size
    first_train_len = max(min_train_size, train_size or 0)
    test_start = first_train_len + gap
    while test_start + test_size <= n:
        train_end = test_start - gap  # exclusive, in unique-time positions
        train_begin = 0 if train_size is None else train_end - train_size
        test_end = test_start + test_size  # exclusive
        train_mask = (inverse >= train_begin) & (inverse < train_end)
        test_mask = (inverse >= test_start) & (inverse < test_end)
        yield Split(
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
            train_start=uniq[train_begin],
            train_end=uniq[train_end - 1],
            test_start=uniq[test_start],
            test_end=uniq[test_end - 1],
        )
        test_start += step


def out_of_time_split(times: npt.ArrayLike, cutoff: Any, *, embargo: Any = None) -> Split:
    """Train on times <= cutoff, test on times > cutoff (+ embargo if given).

    embargo has the same unit as times (e.g. np.timedelta64(12, "M") is not exact; prefer days or
    a cutoff date for the test start). Rows inside the embargo are dropped from both sides.
    """
    arr = _as_array(times)
    test_from = cutoff if embargo is None else cutoff + embargo
    train_mask = arr <= cutoff
    test_mask = arr > test_from
    if not train_mask.any() or not test_mask.any():
        raise ValueError("cutoff leaves an empty train or test set")
    return Split(
        train_idx=np.flatnonzero(train_mask),
        test_idx=np.flatnonzero(test_mask),
        train_start=arr[train_mask].min(),
        train_end=arr[train_mask].max(),
        test_start=arr[test_mask].min(),
        test_end=arr[test_mask].max(),
    )


def purged_kfold(
    times: npt.ArrayLike,
    label_end_times: npt.ArrayLike | None = None,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Split]:
    """Purged, embargoed K-fold over contiguous time blocks (for tuning *inside* a training window only).

    Train rows whose label interval [t, label_end] overlaps the test block are purged; rows that start
    within `embargo` unique time points after the test block are dropped as well. Training data after the
    test block is allowed here by construction; that is why this is for tuning, never for final scores.
    """
    arr = _as_array(times)
    ends = arr if label_end_times is None else _as_array(label_end_times)
    if ends.shape != arr.shape:
        raise ValueError("label_end_times must match times")
    if (ends < arr).any():
        raise ValueError("label_end_times must be >= times")
    if n_splits < 2 or embargo < 0:
        raise ValueError("n_splits must be >= 2, embargo >= 0")

    uniq, inverse = np.unique(arr, return_inverse=True)
    if uniq.size < n_splits:
        raise ValueError("fewer unique time points than folds")
    for block in np.array_split(np.arange(uniq.size), n_splits):
        lo, hi = int(block[0]), int(block[-1])
        t_lo, t_hi = uniq[lo], uniq[hi]
        test_mask = (inverse >= lo) & (inverse <= hi)
        overlaps = (arr <= t_hi) & (ends >= t_lo)
        embargoed = (inverse > hi) & (inverse <= hi + embargo)
        train_mask = ~test_mask & ~overlaps & ~embargoed
        tr = np.flatnonzero(train_mask)
        yield Split(
            train_idx=tr,
            test_idx=np.flatnonzero(test_mask),
            train_start=arr[tr].min() if tr.size else None,
            train_end=arr[tr].max() if tr.size else None,
            test_start=t_lo,
            test_end=t_hi,
        )


def assert_no_lookahead(
    times: npt.ArrayLike,
    split: Split,
    *,
    label_end_times: npt.ArrayLike | None = None,
) -> None:
    """Fail if any training observation (or its label window) reaches the first test time point."""
    arr = _as_array(times)
    ends = arr if label_end_times is None else _as_array(label_end_times)
    if split.train_idx.size == 0 or split.test_idx.size == 0:
        raise LookAheadError("empty train or test set")
    if np.intersect1d(split.train_idx, split.test_idx).size:
        raise LookAheadError("train and test share rows")
    first_test = arr[split.test_idx].min()
    last_train_info = ends[split.train_idx].max()
    if not last_train_info < first_test:
        raise LookAheadError(f"training information reaches {last_train_info!r}, test starts {first_test!r}")
