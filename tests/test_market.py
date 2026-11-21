"""Module B: portfolio construction, VaR/ES models, backtests, and the leakage checks (causality and
shuffled labels) on the real forecasting code. Offline, synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risikopegel.leakage import assert_causal_features
from risikopegel.market import backtest as bt
from risikopegel.market.data import asset_log_returns, portfolio_returns
from risikopegel.market.ml import lgbm_quantile_forecasts, lgbm_vol_forecasts, ml_features
from risikopegel.market.var_models import MODELS, gpd_tail, rolling_forecasts


def garch_series(n: int, seed: int = 0) -> np.ndarray:
    """GARCH(1,1) with t(5) innovations, in percent."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.02, 0.08, 0.9
    s2, out = omega / (1 - alpha - beta), np.empty(n)
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    for t in range(n):
        out[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * out[t] ** 2 + beta * s2
    return out


def test_asset_returns_signs() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    lv = pd.DataFrame(
        {"fx_usd": [1.0, 1.1, 1.1], "fx_gbp": 1.0, "fx_jpy": 1.0, "fx_chf": 1.0,
         "yc_aaa_2y": [1.0, 1.0, 1.0], "yc_aaa_10y": [1.0, 1.1, 1.1]},
        index=idx,
    )  # fmt: skip
    lr = asset_log_returns(lv)
    assert lr["fx_usd"].iloc[0] == pytest.approx(-np.log(1.1))  # EUR appreciates -> USD holding loses
    assert lr["bond_10y"].iloc[0] == pytest.approx(-10 * 0.001 + 0.01 / 252)  # yield up 10bp -> loss
    assert lr["bond_2y"].iloc[1] == pytest.approx(0.01 / 252)  # unchanged yield -> carry only
    with pytest.raises(ValueError, match="sum to 1"):
        portfolio_returns(lr, {"fx_usd": 0.5})


def test_normal_var_matches_closed_form() -> None:
    r = np.random.default_rng(1).normal(0, 1, 200_000)
    f = MODELS["normal"](r, [0.99], None)  # type: ignore[arg-type]
    assert f[0.99][0] == pytest.approx(stats.norm.ppf(0.99), abs=0.02)
    assert f[0.99][1] == pytest.approx(stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, abs=0.02)


def test_gpd_tail_on_exponential() -> None:
    # exponential losses: GPD with xi=0, so VaR_a = -ln(1-a), ES = VaR + 1
    x = np.random.default_rng(2).exponential(1.0, 400_000)
    var, es = gpd_tail(x, 0.99)
    assert var == pytest.approx(np.log(100), rel=0.02)
    assert es == pytest.approx(np.log(100) + 1, rel=0.03)


@pytest.mark.parametrize("model", list(MODELS))
def test_models_es_above_var(model: str) -> None:
    r = garch_series(1200)
    f = rolling_forecasts(r, model, window=1000, levels=[0.99, 0.975], start=1000, stop=1005)
    for a in (0.99, 0.975):
        assert np.all(f["es"][a] >= f["var"][a] - 1e-9)
        assert np.all(f["var"][a] > 0)


@pytest.mark.leakage
@pytest.mark.parametrize("model", ["historical", "ewma", "garch_t", "evt_garch_t"])
def test_rolling_forecasts_are_causal(model: str) -> None:
    """Forecast for day t must not change when everything from t on is replaced."""
    r = garch_series(1100)
    full = rolling_forecasts(r, model, window=1000, levels=[0.99], start=1000, stop=1060)
    tampered = r.copy()
    tampered[1030:] = -50.0  # huge future losses (loss = -return)
    part = rolling_forecasts(tampered, model, window=1000, levels=[0.99], start=1000, stop=1060)
    np.testing.assert_allclose(full["var"][0.99][:31], part["var"][0.99][:31])
    assert not np.allclose(full["var"][0.99][31:], part["var"][0.99][31:])


@pytest.mark.leakage
def test_ml_features_are_causal() -> None:
    r = pd.Series(garch_series(600), index=pd.RangeIndex(600))
    df = pd.DataFrame({"t": np.arange(600), "r": r})

    def fn(d: pd.DataFrame) -> pd.DataFrame:
        return ml_features(d["r"])

    assert_causal_features(fn, df, time_col="t")
    assert ml_features(r).iloc[300].notna().all()
    assert ml_features(r)["r_lag1"].iloc[300] == r.iloc[299]  # row t sees t-1, never t


@pytest.mark.leakage
def test_lgbm_shuffled_labels_lose_the_volatility_signal() -> None:
    """On GARCH data the real model beats unconditional HS; with permuted targets it must not."""
    r = garch_series(3000, seed=3)
    a, start = 0.95, 1000  # 1% quantiles from 1000 points are too noisy for a clear signal
    loss = -r[start:]
    real = lgbm_quantile_forecasts(r, window=1000, levels=[a], start=start, refit_every=50)[a]
    shuf = lgbm_quantile_forecasts(
        r, window=1000, levels=[a], start=start, refit_every=50, shuffle_labels=True
    )[a]
    hs = rolling_forecasts(r, "historical", window=1000, levels=[a], start=start)["var"][a]
    p_real, p_shuf, p_hs = (bt.pinball(loss, v, a).mean() for v in (real, shuf, hs))
    assert p_real < 0.985 * p_hs
    assert p_shuf > 0.99 * p_hs


def test_kupiec_and_christoffersen() -> None:
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True  # exactly 1%, evenly spread
    k = bt.kupiec(hits, 0.01)
    assert k["exceptions"] == 10
    assert k["lr_uc"] == pytest.approx(0.0, abs=1e-9)
    c = bt.christoffersen(hits, 0.01)
    assert c["n11"] == 0
    assert c["p_ind"] > 0.5
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True
    assert bt.christoffersen(clustered, 0.01)["p_ind"] < 1e-6
    assert bt.kupiec(np.ones(100, dtype=bool) & (np.arange(100) < 10), 0.01)["p_uc"] < 1e-4


def test_traffic_light_matches_basel_table() -> None:
    # BCBS MAR32: 250 obs at 99%: green 0-4, amber 5-9, red 10+
    assert [bt.traffic_light_zone(k) for k in (0, 4, 5, 9, 10)] == ["green", "green", "amber", "amber", "red"]
    assert bt.zone_counts(250, 0.01) == {"amber_from": 5, "red_from": 10}


def test_z2_and_scoring_functions() -> None:
    rng = np.random.default_rng(4)
    n, a = 200_000, 0.975
    loss = rng.normal(0, 1, n)
    q = stats.norm.ppf(a)
    var = np.full(n, q)
    es = np.full(n, stats.norm.pdf(q) / (1 - a))
    assert bt.acerbi_szekely_z2(loss, var, es, a) == pytest.approx(0.0, abs=0.03)
    assert bt.acerbi_szekely_z2(loss, var * 0.7, es * 0.7, a) < -0.3  # risk underestimated -> negative
    # the true quantile minimises expected pinball loss and FZ0 vs a mis-scaled forecast
    assert bt.pinball(loss, var, a).mean() < bt.pinball(loss, var * 1.3, a).mean()
    assert bt.fz0(loss, var, es, a).mean() < bt.fz0(loss, var * 0.8, es * 0.8, a).mean()
    dm = bt.diebold_mariano(bt.fz0(loss, var, es, a), bt.fz0(loss, var * 0.8, es * 0.8, a))
    assert dm["stat"] < 0
    assert dm["p_two_sided"] < 0.01


@pytest.mark.leakage
def test_lgbm_vol_fhs_is_causal_and_coherent() -> None:
    r = garch_series(1400, seed=5)
    full = lgbm_vol_forecasts(r, window=1000, levels=[0.99], start=1000, refit_every=20)
    tampered = r.copy()
    tampered[1100:] = -50.0
    part = lgbm_vol_forecasts(tampered, window=1000, levels=[0.99], start=1000, refit_every=20)
    np.testing.assert_allclose(full["var"][0.99][:100], part["var"][0.99][:100])
    assert np.all(full["es"][0.99] >= full["var"][0.99])
