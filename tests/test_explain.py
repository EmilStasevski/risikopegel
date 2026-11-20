"""Additivity of every explanation method: base + sum(contributions) == logit(PD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risikopegel.credit.challengers import (
    fit_catboost,
    fit_ebm,
    fit_lightgbm,
    fit_scorecard,
    fit_xgboost,
)
from risikopegel.credit.explain import explain, logit, reason_agreement, reason_codes


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "k": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    eta = -2 + 1.2 * X["a"] - 0.8 * X["b"] + 0.5 * (X["k"] == "x")
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return X.iloc[:3000], y[:3000], X.iloc[3000:].reset_index(drop=True), y[3000:]


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "catboost", "ebm", "scorecard"])
def test_additivity(name: str, data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    Xtr, ytr, Xte, yte = data
    cat = ["k"]
    if name == "lightgbm":
        f = fit_lightgbm({"num_leaves": 7, "n_estimators": 300}, Xtr, ytr, Xte, yte, seed=0)
    elif name == "xgboost":
        # early stopping must trigger well before n_estimators to exercise the iteration_range fix
        f = fit_xgboost(
            {"max_depth": 6, "learning_rate": 0.3, "early_stopping_rounds": 5}, Xtr, ytr, Xte, yte, seed=0
        )
        assert f.best_iteration is not None
        assert f.best_iteration + 1 < f.model.get_booster().num_boosted_rounds()
    elif name == "catboost":
        f = fit_catboost({"depth": 3, "iterations": 200}, Xtr, ytr, Xte, yte, seed=0, categorical=cat)
    elif name == "ebm":
        f = fit_ebm({"interactions": 2}, Xtr, ytr, Xte, yte, seed=0)
    else:
        f = fit_scorecard({}, Xtr, ytr, Xte, yte, seed=0, categorical=cat)
    e = explain(name, f.model, Xte, cat)
    err = np.abs(e.reconstructed_logit() - logit(f.predict(Xte)))
    assert err.max() < 1e-4, (name, err.max())


def test_reason_codes_and_agreement() -> None:
    c = pd.DataFrame({"a": [0.5, -0.1], "b": [0.2, 0.3], "c": [-0.4, 0.1]})
    assert reason_codes(c, k=2) == [["a", "b"], ["b", "c"]]
    ref = pd.Series({"a": 0.4, "b": 0.0, "c": 0.0})
    assert reason_codes(c, k=3, reference=ref) == [["b", "a"], ["b", "c"]]
    agree = reason_agreement([["a & b", "c"]], [["b"]])
    assert agree["top1_shared"] == 1.0
    assert agree["mean_jaccard_top3"] == pytest.approx(1 / 3)
