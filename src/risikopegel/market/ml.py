"""ML challenger: LightGBM quantile regression for one-day VaR (no ES: a single quantile has none)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

Arr = npt.NDArray[np.float64]

PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "verbose": -1,
}


def ml_features(r: pd.Series) -> pd.DataFrame:
    """Features for the return on day t, built from returns up to t-1 only (every column is shifted)."""
    past = r.shift(1)
    f = pd.DataFrame(index=r.index)
    f["r_lag1"] = past
    f["abs_lag1"] = past.abs()
    for w in (5, 20, 60, 250):
        f[f"sd_{w}"] = past.rolling(w, min_periods=w).std()
    f["ewma_sd"] = np.sqrt((past**2).ewm(alpha=0.06, adjust=False, min_periods=30).mean())
    f["min_20"] = past.rolling(20, min_periods=20).min()
    f["max_abs_20"] = past.abs().rolling(20, min_periods=20).max()
    f["sd_ratio"] = f["sd_5"] / f["sd_60"]
    return f


def lgbm_quantile_forecasts(
    r: Arr,
    *,
    window: int,
    levels: Sequence[float],
    start: int,
    refit_every: int,
    seed: int = 0,
    shuffle_labels: bool = False,
) -> dict[float, Arr]:
    """VaR (positive loss) for each day t >= start, models refitted every `refit_every` days on the
    previous `window` days. shuffle_labels permutes the training targets (leakage check only)."""
    s = pd.Series(r)
    X = ml_features(s).to_numpy()
    rng = np.random.default_rng(seed)
    out = {a: np.full(r.size - start, np.nan) for a in levels}
    models: dict[float, lgb.LGBMRegressor] = {}
    for i, t in enumerate(range(start, r.size)):
        if i % refit_every == 0:
            lo = t - window
            ok = ~np.isnan(X[lo:t]).any(axis=1)
            Xtr, ytr = X[lo:t][ok], r[lo:t][ok]
            if shuffle_labels:
                ytr = rng.permutation(ytr)
            for a in levels:
                m = lgb.LGBMRegressor(objective="quantile", alpha=1 - a, random_state=seed, **PARAMS)
                models[a] = m.fit(Xtr, ytr)
        for a in levels:
            out[a][i] = -float(models[a].predict(X[t : t + 1])[0])
    return out


VOL_PARAMS: dict[str, Any] = {**PARAMS, "n_estimators": 150, "num_leaves": 4, "min_child_samples": 100}


def lgbm_vol_forecasts(
    r: Arr, *, window: int, levels: Sequence[float], start: int, refit_every: int, seed: int = 0
) -> dict[str, dict[float, Arr]]:
    """ML-filtered historical simulation: LightGBM predicts the scale E|r_t| from the same causal features,
    the VaR/ES come from the empirical tail of r / scale in the training window (FHS with an ML filter).
    A fairer ML challenger than a direct 1% quantile, which has only ~10 tail points per 1000 days."""
    s = pd.Series(r)
    X = ml_features(s).to_numpy()
    var = {a: np.full(r.size - start, np.nan) for a in levels}
    es = {a: np.full(r.size - start, np.nan) for a in levels}
    z = np.empty(0)
    m: lgb.LGBMRegressor | None = None
    for i, t in enumerate(range(start, r.size)):
        if i % refit_every == 0:
            lo = t - window
            ok = ~np.isnan(X[lo:t]).any(axis=1)
            Xtr, ytr = X[lo:t][ok], r[lo:t][ok]
            m = lgb.LGBMRegressor(objective="l2", random_state=seed, **VOL_PARAMS).fit(Xtr, np.abs(ytr))
            scale_in = np.maximum(m.predict(Xtr), 1e-6)
            z = -ytr / scale_in  # standardized losses
        assert m is not None
        sc = max(float(m.predict(X[t : t + 1])[0]), 1e-6)
        for a in levels:
            q = float(np.quantile(z, a))
            var[a][i], es[a][i] = sc * q, sc * float(z[z >= q].mean())
    return {"var": var, "es": es}
