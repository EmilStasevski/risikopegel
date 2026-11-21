"""Forecasters for monthly macro series: baselines, ETS, ARIMA, a global LightGBM, foundation models.

Every forecaster returns quantiles (0.1, 0.5, 0.9) for horizons 1..H from a history that ends at the
forecast origin. Nothing after the origin is passed in (the driver slices the history).
Classical model specifications are re-selected once a year (January origins) on data up to that origin and
re-estimated at every origin, which keeps the run affordable without using future data.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

Arr = npt.NDArray[np.float64]
QUANTILES = (0.1, 0.5, 0.9)
H = 12
Z80 = float(stats.norm.ppf(0.9))


def naive(y: Arr, h: int = H) -> Arr:
    return np.repeat(np.full((h, 1), y[-1]), 3, axis=1) + _empirical_spread(np.diff(y), h, seasonal=1)


def seasonal_naive(y: Arr, h: int = H, m: int = 12) -> Arr:
    pt = np.array([y[-m + (i % m)] for i in range(h)])
    return pt[:, None] + _empirical_spread(y[m:] - y[:-m], h, seasonal=m)


def _empirical_spread(d: Arr, h: int, seasonal: int) -> Arr:
    """Quantile offsets for random-walk-type baselines: 10/90% of past one-step (or one-season) changes,
    scaled by sqrt(number of steps) (Hyndman & Athanasopoulos, fpp3 5.5); the median offset is 0."""
    lo, hi = np.quantile(d, [0.1, 0.9]) - np.median(d)
    k = np.array([np.ceil((i + 1) / seasonal) for i in range(h)])
    return np.column_stack([lo * np.sqrt(k), np.zeros(h), hi * np.sqrt(k)])


@dataclass
class SpecCache:
    """Per-series model specification chosen at the last January origin."""

    spec: dict[str, Any] = field(default_factory=dict)


def _ets_fit(y: Arr, trend: bool, seasonal: bool) -> Any:
    mod = ETSModel(
        pd.Series(y),
        error="add",
        trend="add" if trend else None,
        damped_trend=trend,
        seasonal="add" if seasonal else None,
        seasonal_periods=12 if seasonal else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=500)


def ets(y: Arr, cache: SpecCache, key: str, reselect: bool, h: int = H) -> Arr:
    if reselect or key not in cache.spec:
        fits = {(t, s): _ets_fit(y, t, s) for t, s in itertools.product([False, True], [False, True])}
        cache.spec[key] = min(fits, key=lambda k: fits[k].aicc)
        res = fits[cache.spec[key]]
    else:
        res = _ets_fit(y, *cache.spec[key])
    f = res.get_prediction(start=y.size, end=y.size + h - 1).summary_frame(alpha=0.2)
    return np.column_stack([f["pi_lower"].to_numpy(), f["mean"].to_numpy(), f["pi_upper"].to_numpy()])


def _arima_fit(y: Arr, order: tuple[int, int, int], seasonal: bool, trend: str) -> Any:
    so = (0, 1, 1, 12) if seasonal else (0, 0, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, order=order, seasonal_order=so, trend=trend).fit(disp=False, maxiter=200)


def arima(y: Arr, cache: SpecCache, key: str, reselect: bool, seasonal: bool, h: int = H) -> Arr:  # noqa: PLR0917
    """ARIMA(p,1,q) with optional drift; seasonal series get an airline-type (0,1,1)_12 part.
    p, q in {0,1,2} and drift chosen by AIC at January origins."""
    if reselect or key not in cache.spec:
        best, best_aic = None, np.inf
        for p, q, tr in itertools.product(range(3), range(3), ["n", "t"]):
            try:
                r = _arima_fit(y, (p, 1, q), seasonal, tr)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(r.aic) and r.aic < best_aic:
                best, best_aic = ((p, 1, q), tr), r.aic
        cache.spec[key] = best or ((0, 1, 1), "n")
    order, tr = cache.spec[key]
    res = _arima_fit(y, order, seasonal, tr)
    fc = res.get_forecast(h)
    ci = np.asarray(fc.conf_int(alpha=0.2))
    return np.column_stack([ci[:, 0], np.asarray(fc.predicted_mean), ci[:, 1]])


# ---------------------------------------------------------------- global LightGBM (direct, h in HORIZONS)
LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": 4,
}
N_LAGS = 12


def lgb_features(y: Arr, scale: float, month: npt.NDArray[np.int_]) -> Arr:
    """Row t: last N_LAGS scaled first differences up to t, level vs 12-month mean, calendar month of t."""
    d = np.diff(y, prepend=np.nan) / scale
    cols = [np.roll(d, k) for k in range(N_LAGS)]
    for k, c in enumerate(cols):
        c[:k] = np.nan
    lvl = (y - pd.Series(y).rolling(12).mean().to_numpy()) / scale
    return np.column_stack([*cols, lvl, month])


def lgb_train_rows(series: dict[str, tuple[Arr, npt.NDArray[np.int_]]], h: int) -> tuple[Arr, Arr]:
    X, Y = [], []
    for y, month in series.values():
        scale = float(np.nanstd(np.diff(y))) or 1.0
        F = lgb_features(y, scale, month)
        tgt = np.full(y.size, np.nan)
        tgt[: y.size - h] = (y[h:] - y[:-h]) / scale  # only pairs fully inside the history
        ok = ~np.isnan(F).any(axis=1) & ~np.isnan(tgt)
        X.append(F[ok])
        Y.append(tgt[ok])
    return np.vstack(X), np.concatenate(Y)


def lgb_fit(
    series: dict[str, tuple[Arr, npt.NDArray[np.int_]]], h: int, seed: int = 0
) -> list[lgb.LGBMRegressor]:
    X, Y = lgb_train_rows(series, h)
    return [
        lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed, **LGB_PARAMS).fit(X, Y)
        for q in QUANTILES
    ]


def lgb_predict(models: list[lgb.LGBMRegressor], y: Arr, month: npt.NDArray[np.int_]) -> Arr:
    scale = float(np.nanstd(np.diff(y))) or 1.0
    x = lgb_features(y, scale, month)[-1:]
    q = np.sort([float(m.predict(x)[0]) for m in models])  # no quantile crossing
    return np.asarray(y[-1] + q * scale, dtype=np.float64)
