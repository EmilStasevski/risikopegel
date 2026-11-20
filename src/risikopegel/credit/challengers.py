"""Challenger PD models against the WoE scorecard (PLAN §3). One interface for all of them.

Every model is fitted on train, uses valid only for early stopping / model selection, and is scored on
test exactly once by the calling script.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from interpret.glassbox import ExplainableBoostingClassifier
from optbinning import BinningProcess, Scorecard
from scipy import stats
from sklearn.linear_model import LogisticRegression

Predict = Callable[[pd.DataFrame], npt.NDArray[np.float64]]


@dataclass
class Fitted:
    name: str
    params: dict[str, Any]
    predict: Predict
    model: Any
    fit_seconds: float
    best_iteration: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def grid(space: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def monotone_signs(
    X: pd.DataFrame, y: npt.ArrayLike, numeric: Sequence[str], min_abs_rho: float = 0.02
) -> dict[str, int]:
    """Constraint direction per numeric feature from the train-set Spearman correlation with default.

    Weak correlations (|rho| < min_abs_rho) stay unconstrained (0). Train data only, so no leakage.
    """
    ya = np.asarray(y)
    out: dict[str, int] = {}
    for c in numeric:
        m = X[c].notna().to_numpy()
        if m.sum() < 100 or X.loc[m, c].nunique() < 2:
            out[c] = 0
            continue
        rho = float(stats.spearmanr(X.loc[m, c], ya[m]).statistic)
        out[c] = 0 if abs(rho) < min_abs_rho else int(np.sign(rho))
    return out


# ---------------------------------------------------------------- LightGBM
def fit_lightgbm(
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: npt.ArrayLike,
    X_va: pd.DataFrame,
    y_va: npt.ArrayLike,
    *,
    seed: int,
    monotone: Mapping[str, int] | None = None,
) -> Fitted:
    p = {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "random_state": seed,
        "n_jobs": 16,
        "verbose": -1,
        **params,
    }
    if monotone is not None:
        p["monotone_constraints"] = [monotone.get(c, 0) for c in X_tr.columns]
        p["monotone_constraints_method"] = "advanced"
    t0 = time.perf_counter()
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, np.asarray(y_tr), eval_X=(X_va,), eval_y=(np.asarray(y_va),), eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])  # fmt: skip
    dt = time.perf_counter() - t0
    name = "lightgbm_monotone" if monotone is not None else "lightgbm"
    return Fitted(
        name,
        params,
        lambda X: np.asarray(m.predict_proba(X))[:, 1].astype(np.float64),
        m,
        dt,
        int(m.best_iteration_),
    )


# ---------------------------------------------------------------- CatBoost
def fit_catboost(
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: npt.ArrayLike,
    X_va: pd.DataFrame,
    y_va: npt.ArrayLike,
    *,
    seed: int,
    categorical: Sequence[str],
) -> Fitted:
    def pool(X: pd.DataFrame, y: npt.ArrayLike | None = None) -> Pool:
        Xc = X.copy()
        for c in categorical:
            Xc[c] = Xc[c].astype(str)
        return Pool(Xc, y, cat_features=list(categorical))

    p = {
        "iterations": 5000,
        "learning_rate": 0.05,
        "eval_metric": "AUC",
        "random_seed": seed,
        "thread_count": 16,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 200,
        "allow_writing_files": False,
        **params,
    }
    t0 = time.perf_counter()
    m = CatBoostClassifier(**p)
    m.fit(pool(X_tr, y_tr), eval_set=pool(X_va, y_va), use_best_model=True)
    dt = time.perf_counter() - t0
    return Fitted(
        "catboost", params, lambda X: m.predict_proba(pool(X))[:, 1].astype(np.float64), m, dt,
        int(m.get_best_iteration()),
    )  # fmt: skip


# ---------------------------------------------------------------- XGBoost
def fit_xgboost(
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: npt.ArrayLike,
    X_va: pd.DataFrame,
    y_va: npt.ArrayLike,
    *,
    seed: int,
) -> Fitted:
    p = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 1,
        "eval_metric": "auc",
        "early_stopping_rounds": 200,
        "random_state": seed,
        "n_jobs": 16,
        **params,
    }
    t0 = time.perf_counter()
    m = xgb.XGBClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    dt = time.perf_counter() - t0
    return Fitted(
        "xgboost", params, lambda X: m.predict_proba(X)[:, 1].astype(np.float64), m, dt, int(m.best_iteration)
    )


# ---------------------------------------------------------------- EBM (glass box)
def fit_ebm(
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: npt.ArrayLike,
    X_va: pd.DataFrame,
    y_va: npt.ArrayLike,
    *,
    seed: int,
) -> Fitted:
    # EBM does its own internal validation split for early stopping; X_va is not used for fitting.
    p = {"random_state": seed, "n_jobs": 16, **params}
    t0 = time.perf_counter()
    m = ExplainableBoostingClassifier(**p)
    m.fit(X_tr, y_tr)
    dt = time.perf_counter() - t0
    return Fitted("ebm", params, lambda X: m.predict_proba(X)[:, 1].astype(np.float64), m, dt)


# ---------------------------------------------------------------- WoE scorecard
def fit_scorecard(
    params: dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: npt.ArrayLike,
    X_va: pd.DataFrame,
    y_va: npt.ArrayLike,
    *,
    seed: int,
    categorical: Sequence[str],
) -> Fitted:
    feats = list(X_tr.columns)
    fit_params: dict[str, dict[str, Any]] = {
        f: {"monotonic_trend": "auto_asc_desc"} for f in feats if f not in categorical
    }
    bp = BinningProcess(
        variable_names=feats,
        categorical_variables=list(categorical),
        max_n_prebins=20,
        min_bin_size=0.05,
        selection_criteria={"iv": {"min": params.get("min_iv", 0.02)}},
        binning_fit_params=fit_params,
        n_jobs=16,
    )
    sc = Scorecard(
        binning_process=bp,
        estimator=LogisticRegression(C=params.get("C", 1.0), max_iter=2000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
    )

    def to_obj(X: pd.DataFrame) -> pd.DataFrame:
        Xo = X.copy()
        for c in categorical:
            Xo[c] = Xo[c].astype(str)
        return Xo

    t0 = time.perf_counter()
    sc.fit(to_obj(X_tr), np.asarray(y_tr))
    dt = time.perf_counter() - t0
    return Fitted(
        "scorecard", params, lambda X: sc.predict_proba(to_obj(X))[:, 1].astype(np.float64), sc, dt,
        extra={"n_selected": len(sc.binning_process_.get_support(names=True))},
    )  # fmt: skip
