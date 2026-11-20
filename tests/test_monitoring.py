from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risikopegel.credit.challengers import grid, monotone_signs
from risikopegel.credit.monitoring import csi_table, psi, psi_categorical


def test_psi_zero_for_same_distribution_and_grows_with_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=50_000)
    assert psi(a, rng.normal(size=50_000)) < 0.002
    small, large = psi(a, rng.normal(0.2, 1, 50_000)), psi(a, rng.normal(1.0, 1, 50_000))
    assert 0.02 < small < 0.1
    assert large > 0.5


def test_psi_known_value_two_bins() -> None:
    # expected 50/50 split, actual 70/30 -> (0.2*ln(1.4) + (-0.2)*ln(0.6)) = 0.1695...
    e = np.array([0.0] * 50 + [1.0] * 50)
    a = np.array([0.0] * 70 + [1.0] * 30)
    assert psi(e, a, n_bins=2) == pytest.approx(0.2 * np.log(1.4) - 0.2 * np.log(0.6), rel=1e-9)


def test_psi_treats_missing_as_own_bin() -> None:
    e = np.r_[np.arange(100.0), [np.nan] * 0]
    a = np.r_[np.arange(100.0), [np.nan] * 100]
    assert psi(e, a) > 0.5


def test_psi_categorical_and_csi_table() -> None:
    e = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(e, e) == pytest.approx(0.0)
    assert psi_categorical(e, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.3
    exp = pd.DataFrame({"x": np.arange(1000.0), "c": ["a", "b"] * 500})
    act = pd.DataFrame({"x": np.arange(1000.0) + 500, "c": ["a", "b"] * 500})
    t = csi_table(exp, act)
    assert t["feature"].tolist() == ["x", "c"]


def test_monotone_signs_from_train_correlation() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    X = pd.DataFrame({"up": rng.normal(size=n), "down": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = rng.binomial(1, 1 / (1 + np.exp(-(X["up"] - X["down"]))))
    s = monotone_signs(X, y, ["up", "down", "noise"], min_abs_rho=0.05)
    assert s == {"up": 1, "down": -1, "noise": 0}


def test_grid() -> None:
    assert list(grid({"a": [1, 2], "b": ["x"]})) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
