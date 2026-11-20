from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from risikopegel.credit.metrics import (
    calibration_slope_intercept,
    calibration_table,
    ks_statistic,
    pd_metrics,
)
from risikopegel.credit.scorecard import ScorecardConfig, fit_predict_pd
from risikopegel.leakage import shuffled_label_check
from risikopegel.splits import Split


def _calibrated(n: int = 50_000, seed: int = 0) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-rng.normal(-2.5, 1.0, size=n)))
    return rng.binomial(1, p), p


def test_calibrated_scores_have_slope_one_intercept_zero() -> None:
    y, p = _calibrated()
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_overconfident_scores_have_slope_below_one() -> None:
    y, p = _calibrated()
    logit = np.log(p / (1 - p))
    p_over = 1 / (1 + np.exp(-(2 * logit - np.log(p.mean() / (1 - p.mean())))))
    slope, _ = calibration_slope_intercept(y, p_over)
    assert slope == pytest.approx(0.5, abs=0.05)


def test_ks_and_gini_on_perfect_and_random_scores() -> None:
    y = np.array([0, 0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    m = pd_metrics(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert m["gini"] == 1.0
    assert m["auc"] == 1.0


def test_calibration_table_buckets() -> None:
    y, p = _calibrated(10_000)
    t = calibration_table(y, p, n_bins=10)
    assert len(t) == 10
    assert t["n"].sum() == 10_000
    assert t["mean_pd"].is_monotonic_increasing
    assert (t["abs_gap"] < 0.03).all()


def _synthetic_applicants(n: int = 6000, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    util = rng.uniform(0, 1.2, n)
    age = rng.integers(21, 80, n).astype(float)
    late = rng.poisson(0.3, n).astype(float)
    late[rng.choice(n, 20, replace=False)] = 98.0  # special code
    eta = -3 + 2.0 * util - 0.02 * (age - 45) + 0.8 * np.minimum(late, 5)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return pd.DataFrame({"util": util, "age": age, "late": late}), y


def test_scorecard_end_to_end_on_synthetic_data() -> None:
    X, y = _synthetic_applicants()
    cfg = ScorecardConfig(special_codes={"late": [98]})
    feats = ["util", "age", "late"]
    sc, p = fit_predict_pd(feats, cfg, X.iloc[:4000], y[:4000], X.iloc[4000:])
    assert roc_auc_score(y[4000:], p) > 0.7
    # monotonic bins: utilisation WoE must be monotone
    tbl = sc.binning_process_.get_binned_variable("util").binning_table.build()
    woe = tbl.loc[tbl["Count"] > 0, "WoE"].iloc[:-3]  # drop Special/Missing/Totals rows
    woe = pd.to_numeric(woe[woe != ""], errors="coerce").dropna()
    assert woe.is_monotonic_increasing or woe.is_monotonic_decreasing
    points = sc.table(style="summary")
    assert {"Variable", "Bin", "Points"} <= set(points.columns)


@pytest.mark.leakage
def test_scorecard_pipeline_shuffled_labels_at_chance() -> None:
    X, y = _synthetic_applicants()
    feats = ["util", "age", "late"]
    cfg = ScorecardConfig(special_codes={"late": [98]}, min_iv=0.0)
    split = Split(np.arange(4000), np.arange(4000, 6000), None, None, None, None)

    def fit_score(Xa: pd.DataFrame, ya: Any, Xb: pd.DataFrame, yb: Any) -> float:
        _, pb = fit_predict_pd(feats, cfg, Xa, ya, Xb)
        return float(roc_auc_score(yb, pb))

    res = shuffled_label_check(fit_score, X, y, split, chance=0.5, tolerance=0.02, n_rounds=10)
    assert res.passed, (res.mean, res.se)
