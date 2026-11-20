"""Discrimination and calibration metrics for PD models. Never report AUC alone (PLAN §10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

_EPS = 1e-12


def ks_statistic(y: npt.ArrayLike, p: npt.ArrayLike) -> float:
    """Kolmogorov-Smirnov: max distance between score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
    return float(np.max(tpr - fpr))


def calibration_slope_intercept(y: npt.ArrayLike, p: npt.ArrayLike) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: slope 1, intercept 0."""
    pa = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    logit = np.log(pa / (1 - pa)).reshape(-1, 1)
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(logit, np.asarray(y))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def calibration_table(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Equal-frequency buckets of predicted PD with observed default rate per bucket."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = df.groupby("bucket").agg(n=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean"))
    out["abs_gap"] = (out["mean_pd"] - out["default_rate"]).abs()
    return out.reset_index()


def expected_calibration_error(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * t["abs_gap"]).sum() / t["n"].sum())


def pd_metrics(y: npt.ArrayLike, p: npt.ArrayLike) -> dict[str, Any]:
    ya, pa = np.asarray(y), np.asarray(p, dtype=float)
    auc = float(roc_auc_score(ya, pa))
    slope, intercept = calibration_slope_intercept(ya, pa)
    return {
        "n": int(ya.size),
        "default_rate": float(ya.mean()),
        "mean_pd": float(pa.mean()),
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(ya, pa),
        "pr_auc": float(average_precision_score(ya, pa)),
        "brier": float(brier_score_loss(ya, pa)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece": expected_calibration_error(ya, pa),
    }
