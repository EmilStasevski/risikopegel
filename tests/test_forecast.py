"""Module C: forecasters, causality of the LightGBM features and training rows, shuffled-label check,
energy meter. Offline and synthetic (foundation models are not loaded in tests)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from risikopegel.forecast import models as m
from risikopegel.forecast.energy import measure
from risikopegel.leakage import assert_causal_features


def seasonal_series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.asarray(100 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + np.cumsum(rng.normal(0, 0.3, n)))


def test_baselines_shape_and_order() -> None:
    y = seasonal_series(120)
    for q in (m.naive(y), m.seasonal_naive(y)):
        assert q.shape == (12, 3)
        assert np.all(q[:, 0] <= q[:, 1])
        assert np.all(q[:, 1] <= q[:, 2])
    assert m.naive(y)[5, 1] == y[-1]
    assert m.seasonal_naive(y)[0, 1] == y[-12]
    assert m.seasonal_naive(y)[11, 1] == y[-1]


def test_ets_and_arima_pick_up_seasonality() -> None:
    y = seasonal_series(150)
    truth = 100 + 0.1 * np.arange(150, 162) + 3 * np.sin(2 * np.pi * np.arange(150, 162) / 12)
    c = m.SpecCache()
    e = m.ets(y, c, "s", reselect=True)
    assert c.spec["s"][1] is True  # seasonal component selected
    a = m.arima(y, c, "a", reselect=True, seasonal=True)
    flat = np.full(12, y[-1])
    for q in (e, a):
        assert np.mean(np.abs(q[:, 1] - truth)) < np.mean(np.abs(flat - truth))


@pytest.mark.leakage
def test_lgb_features_are_causal() -> None:
    y = seasonal_series(200)
    month = (np.arange(200) % 12) + 1
    df = pd.DataFrame({"t": np.arange(200), "y": y, "month": month})

    def fn(d: pd.DataFrame) -> pd.DataFrame:
        f = m.lgb_features(d["y"].to_numpy(), 1.0, d["month"].to_numpy())
        return pd.DataFrame(f, index=d.index)

    assert_causal_features(fn, df, time_col="t")


@pytest.mark.leakage
def test_lgb_training_rows_never_reach_past_the_history() -> None:
    """Targets y[t+h]-y[t] are built only where t+h is inside the history passed in."""
    y = seasonal_series(100)
    month = (np.arange(100) % 12) + 1
    _, Y = m.lgb_train_rows({"s": (y, month)}, h=12)
    scale = float(np.std(np.diff(y)))
    last_t = 100 - 12 - 1
    assert Y[-1] == pytest.approx((y[last_t + 12] - y[last_t]) / scale)
    assert len(Y) == 100 - 12 - 12  # 12 warm-up rows (lags, 12m mean) and 12 rows without a target


@pytest.mark.leakage
def test_lgb_shuffled_targets_lose_skill() -> None:
    """Global LightGBM on seasonal series beats a flat forecast; trained on permuted targets it must not
    beat it by the same margin (checks that features, not the target, carry the skill)."""
    rng = np.random.default_rng(1)
    series = {f"s{i}": (seasonal_series(240, seed=i), (np.arange(240) % 12) + 1) for i in range(6)}
    hist = {k: (v[0][:228], v[1][:228]) for k, v in series.items()}
    h = 3
    real = m.lgb_fit(hist, h)
    X, Y = m.lgb_train_rows(hist, h)
    shuf = [
        lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=0, **m.LGB_PARAMS).fit(
            X, rng.permutation(Y)
        )
        for q in m.QUANTILES
    ]
    err_real, err_shuf, err_flat = [], [], []
    for y, month in series.values():
        for o in range(228, 237):
            truth = y[o + h - 1]
            err_real.append(abs(m.lgb_predict(real, y[:o], month[:o])[1] - truth))
            err_shuf.append(abs(m.lgb_predict(shuf, y[:o], month[:o])[1] - truth))
            err_flat.append(abs(y[o - 1] - truth))
    assert np.mean(err_real) < 0.8 * np.mean(err_flat)
    assert np.mean(err_shuf) > 0.9 * np.mean(err_flat)


def test_energy_meter_runs() -> None:
    with measure() as e:
        sum(range(100_000))
    assert e.seconds > 0
    assert e.joules is None or e.joules >= 0
