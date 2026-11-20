"""Leakage guards that run in CI (PLAN §0 rule 4).

1. Causality check: a feature pipeline must give the same value for row t whether or not rows after t
   exist. Recompute on data truncated at several cutoffs and compare (catches centred rolling windows,
   missing shift(), full-sample scaling/binning, backward-fill, ...).
2. Timestamp check: every feature's availability time must be <= the decision time of its row.
3. Shuffled-label check: a model trained on permuted training labels must score at chance level on the
   test fold. If it does not, the features or the split encode the label.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from risikopegel.splits import LookAheadError, Split

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


def assert_causal_features(
    feature_fn: FeatureFn,
    df: pd.DataFrame,
    *,
    time_col: str,
    cutoffs: Sequence[Any] | None = None,
    n_cutoffs: int = 5,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Fail if any feature value at time <= cutoff changes when rows after the cutoff are removed.

    feature_fn must return a frame aligned to its input index (same index labels for the kept rows).
    """
    if time_col not in df.columns:
        raise KeyError(time_col)
    full = feature_fn(df)
    if cutoffs is None:
        uniq = np.sort(df[time_col].unique())
        # interior cutoffs only: truncating at the last time point is a no-op
        pos = np.linspace(0, uniq.size - 1, n_cutoffs + 2)[1:-1].astype(int)
        cutoffs = [uniq[p] for p in np.unique(pos)]
    problems: list[str] = []
    for cutoff in cutoffs:
        part = df[df[time_col] <= cutoff]
        trunc = feature_fn(part)
        common_cols = [c for c in trunc.columns if c in full.columns]
        ref = full.loc[trunc.index, common_cols]
        got = trunc[common_cols]
        for col in common_cols:
            a, b = ref[col].to_numpy(), got[col].to_numpy()
            if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                same = np.isclose(a.astype(float), b.astype(float), rtol=rtol, atol=atol, equal_nan=True)
            else:
                same = (a == b) | (pd.isna(a) & pd.isna(b))
            if not same.all():
                problems.append(f"{col!r} (cutoff {cutoff!r}, {int((~same).sum())} rows differ)")
    if problems:
        raise LookAheadError("features depend on future rows: " + "; ".join(sorted(set(problems))))


def assert_feature_timestamps(
    df: pd.DataFrame, *, decision_time_col: str, available_time_cols: Sequence[str]
) -> None:
    """Fail if any feature became available after the decision it is used for."""
    bad = {
        col: int((df[col] > df[decision_time_col]).sum())
        for col in available_time_cols
        if (df[col] > df[decision_time_col]).any()
    }
    if bad:
        raise LookAheadError(f"features available after decision time (rows per column): {bad}")


@dataclass(frozen=True)
class ShuffledLabelResult:
    scores: npt.NDArray[np.float64]
    chance: float
    tolerance: float
    z: float = 3.0

    @property
    def mean(self) -> float:
        return float(self.scores.mean())

    @property
    def sd(self) -> float:
        return float(self.scores.std(ddof=1)) if self.scores.size > 1 else 0.0

    @property
    def se(self) -> float:
        return self.sd / float(np.sqrt(self.scores.size))

    @property
    def passed(self) -> bool:
        # A model fitted on noise still ranks by *some* feature combination; with a few strong features
        # its test AUC scatters widely around chance in both directions (GMSC scorecard: SD ~0.10 over
        # 20 rounds). A leak shifts every round the same way. So: mean within max(tolerance, z * SE).
        return abs(self.mean - self.chance) <= max(self.tolerance, self.z * self.se)


FitScore = Callable[[pd.DataFrame, npt.NDArray[Any], pd.DataFrame, npt.NDArray[Any]], float]


def shuffled_label_check(
    fit_score: FitScore,
    X: pd.DataFrame,
    y: npt.ArrayLike,
    split: Split,
    *,
    chance: float,
    tolerance: float,
    n_rounds: int = 20,
    seed: int = 0,
    z: float = 3.0,
) -> ShuffledLabelResult:
    """Train on permuted training labels (test labels untouched) and score on the test fold.

    fit_score(X_train, y_train, X_test, y_test) -> metric, e.g. AUC (chance 0.5). It must wrap the
    *whole* pipeline (encoding, binning, scaling, model), because that is where labels leak: a step that
    peeks at y_test or at full-sample statistics keeps scoring above chance after the permutation.
    """
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    X_tr, X_te = X.iloc[split.train_idx], X.iloc[split.test_idx]
    y_tr, y_te = y_arr[split.train_idx], y_arr[split.test_idx]
    scores = np.array(
        [fit_score(X_tr, rng.permutation(y_tr), X_te, y_te) for _ in range(n_rounds)], dtype=np.float64
    )
    return ShuffledLabelResult(scores=scores, chance=chance, tolerance=tolerance, z=z)
