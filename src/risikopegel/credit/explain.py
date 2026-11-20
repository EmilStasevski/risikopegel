"""Local explanations in log-odds space for every model family, plus reason codes.

Exact methods only (no sampling):
- LightGBM / XGBoost / CatBoost: the libraries' built-in TreeSHAP (`pred_contrib` / `pred_contribs` /
  `ShapValues`), i.e. the same algorithm as the `shap` package's TreeExplainer.
- EBM: the model's own additive terms (`eval_terms`), exact by construction.
- WoE scorecard: coefficient x WoE per variable (the logistic model is additive in log-odds).

Every explanation must satisfy additivity: base + sum(contributions) == logit(PD). Checked by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from catboost import Pool


@dataclass(frozen=True)
class Explanation:
    contributions: pd.DataFrame  # rows = applicants, columns = features (or EBM terms), log-odds units
    base: npt.NDArray[np.float64]  # per-row base value (constant for trees/EBM)
    method: str

    def reconstructed_logit(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.base + self.contributions.to_numpy().sum(axis=1), dtype=np.float64)


def logit(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def explain(name: str, model: Any, X: pd.DataFrame, categorical: Sequence[str]) -> Explanation:
    cols = list(X.columns)
    if name.startswith("lightgbm"):
        raw = np.asarray(model.predict(X, pred_contrib=True), dtype=np.float64)
        return Explanation(
            pd.DataFrame(raw[:, :-1], columns=cols), raw[:, -1], "TreeSHAP (LightGBM pred_contrib)"
        )
    if name == "xgboost":
        dm = xgb.DMatrix(X, enable_categorical=True)
        # Booster.predict uses ALL trees by default, but the fitted classifier predicts with the
        # early-stopped prefix. Without iteration_range the SHAP values explain a different model
        # (additivity error up to ~1.7 logits observed). Bug caught by the additivity check.
        rng = (0, int(model.best_iteration) + 1)
        raw = np.asarray(
            model.get_booster().predict(dm, pred_contribs=True, iteration_range=rng), dtype=np.float64
        )
        return Explanation(
            pd.DataFrame(raw[:, :-1], columns=cols), raw[:, -1], "TreeSHAP (XGBoost pred_contribs)"
        )
    if name == "catboost":
        Xc = X.copy()
        for c in categorical:
            Xc[c] = Xc[c].astype(str)
        raw = np.asarray(
            model.get_feature_importance(Pool(Xc, cat_features=list(categorical)), type="ShapValues"),
            dtype=np.float64,
        )
        return Explanation(
            pd.DataFrame(raw[:, :-1], columns=cols), raw[:, -1], "TreeSHAP (CatBoost ShapValues)"
        )
    if name == "ebm":
        terms = np.asarray(model.eval_terms(X), dtype=np.float64)
        base = np.full(len(X), float(np.ravel(model.intercept_)[0]))
        return Explanation(pd.DataFrame(terms, columns=list(model.term_names_)), base, "EBM additive terms")
    if name == "scorecard":
        Xo = X.copy()
        for c in categorical:
            Xo[c] = Xo[c].astype(str)
        bp = model.binning_process_
        sel = list(bp.get_support(names=True))
        woe = bp.transform(Xo, metric="woe")[sel]
        coef = np.ravel(model.estimator_.coef_)
        contrib = woe.to_numpy(dtype=np.float64) * coef
        base = np.full(len(X), float(np.ravel(model.estimator_.intercept_)[0]))
        return Explanation(pd.DataFrame(contrib, columns=sel), base, "coefficient x WoE (logistic, exact)")
    raise ValueError(name)


def term_features(term: str) -> frozenset[str]:
    """EBM interaction terms are named 'a & b'; map a term to the features it involves."""
    return frozenset(t.strip() for t in term.split(" & "))


def reason_codes(contrib: pd.DataFrame, k: int = 3, reference: pd.Series | None = None) -> list[list[str]]:
    """Top-k risk-increasing terms per row.

    reference: per-term value subtracted first (e.g. the best achievable scorecard contribution, so the
    reason is "points lost vs the best bin"). Without it, the raw contribution is used (SHAP convention:
    risk relative to the average applicant). Only positive gaps count as reasons.
    """
    m = contrib.to_numpy(dtype=np.float64)
    if reference is not None:
        m = m - reference[contrib.columns].to_numpy(dtype=np.float64)
    names = np.asarray(contrib.columns)
    order = np.argsort(-m, axis=1)[:, :k]
    out: list[list[str]] = []
    for i, idx in enumerate(order):
        out.append([str(names[j]) for j in idx if m[i, j] > 0])
    return out


def reason_agreement(a: list[list[str]], b: list[list[str]]) -> dict[str, float]:
    """Compare two reason-code lists row by row at feature level (EBM terms expanded)."""
    top1, jacc = [], []
    for ra, rb in zip(a, b, strict=True):
        fa = set().union(*(term_features(t) for t in ra)) if ra else set()
        fb = set().union(*(term_features(t) for t in rb)) if rb else set()
        top1.append(bool(ra and rb and term_features(ra[0]) & term_features(rb[0])))
        jacc.append(len(fa & fb) / len(fa | fb) if fa | fb else 1.0)
    return {
        "n": float(len(a)),
        "top1_shared": float(np.mean(top1)),
        "mean_jaccard_top3": float(np.mean(jacc)),
    }
