from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from risikopegel.leakage import (
    assert_causal_features,
    assert_feature_timestamps,
    shuffled_label_check,
)
from risikopegel.splits import LookAheadError, out_of_time_split

pytestmark = pytest.mark.leakage


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    ret = rng.standard_t(df=4, size=n) * 0.01
    return pd.DataFrame({"date": dates, "ret": ret})


def causal_features(df: pd.DataFrame) -> pd.DataFrame:
    r = df["ret"]
    return pd.DataFrame(
        {
            "lag1": r.shift(1),
            "vol20": r.shift(1).rolling(20).std(),
            "ewma_var": (r.shift(1) ** 2).ewm(alpha=0.06, adjust=False).mean(),
        },
        index=df.index,
    )


def test_causal_features_pass(prices: pd.DataFrame) -> None:
    assert_causal_features(causal_features, prices, time_col="date")


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("centred window", lambda d: pd.DataFrame({"x": d["ret"].rolling(21, center=True).std()})),
        ("lead", lambda d: pd.DataFrame({"x": d["ret"].shift(-1)})),
        ("full-sample z-score", lambda d: pd.DataFrame({"x": (d["ret"] - d["ret"].mean()) / d["ret"].std()})),
        ("backfill", lambda d: pd.DataFrame({"x": d["ret"].where(d.index % 5 == 0).bfill()})),
    ],
)
def test_future_dependent_features_fail(prices: pd.DataFrame, name: str, fn: Any) -> None:
    with pytest.raises(LookAheadError, match="'x'"):
        assert_causal_features(fn, prices, time_col="date")


def test_feature_timestamps() -> None:
    df = pd.DataFrame(
        {
            "decision": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "bureau_pull": pd.to_datetime(["2020-01-30", "2020-02-29"]),
            "macro_release": pd.to_datetime(["2020-01-15", "2020-03-10"]),  # released after decision
        }
    )
    assert_feature_timestamps(df, decision_time_col="decision", available_time_cols=["bureau_pull"])
    with pytest.raises(LookAheadError, match="macro_release"):
        assert_feature_timestamps(
            df, decision_time_col="decision", available_time_cols=["bureau_pull", "macro_release"]
        )


def _credit_like(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, npt.NDArray[np.int_], pd.Series]:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.integers(0, 200, size=n)
    p = 1 / (1 + np.exp(-(-2 + 1.2 * x1 - 0.8 * x2)))
    y = rng.binomial(1, p)
    dates = pd.Series(pd.date_range("2018-01-01", periods=n, freq="6h"))
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y, dates


def _honest_pipeline(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any) -> float:
    m = LogisticRegression().fit(X_tr[["x1", "x2"]], y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(X_te[["x1", "x2"]])[:, 1]))


def _target_encoding_on_full_sample(X_tr: pd.DataFrame, y_tr: Any, X_te: pd.DataFrame, y_te: Any) -> float:
    # Bug under test: the category's default rate is estimated on train AND test labels.
    allx = pd.concat([X_tr, X_te])
    ally = np.concatenate([y_tr, y_te])
    te = pd.Series(ally, index=allx.index).groupby(allx["cat"]).mean()
    f_tr = X_tr.assign(te=X_tr["cat"].map(te))[["te"]]
    f_te = X_te.assign(te=X_te["cat"].map(te))[["te"]]
    m = LogisticRegression().fit(f_tr, y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(f_te)[:, 1]))


def test_shuffled_labels_give_chance_for_honest_pipeline() -> None:
    X, y, dates = _credit_like()
    split = out_of_time_split(dates.to_numpy(), dates.iloc[2999])
    honest = _honest_pipeline(
        X.iloc[split.train_idx], y[split.train_idx], X.iloc[split.test_idx], y[split.test_idx]
    )
    assert honest > 0.7  # the signal is real
    res = shuffled_label_check(_honest_pipeline, X, y, split, chance=0.5, tolerance=0.05, n_rounds=10)
    assert res.passed, res.scores


def test_shuffled_labels_expose_target_leak() -> None:
    X, y, dates = _credit_like()
    split = out_of_time_split(dates.to_numpy(), dates.iloc[2999])
    res = shuffled_label_check(
        _target_encoding_on_full_sample, X, y, split, chance=0.5, tolerance=0.05, n_rounds=10
    )
    assert not res.passed, f"leak not detected, mean AUC {res.mean:.3f}"
