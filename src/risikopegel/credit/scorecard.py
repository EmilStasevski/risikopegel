"""WoE/logistic scorecard (the baseline to beat and to respect, PLAN §3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ScorecardConfig:
    # Points scaling: `base_points` at good:bad odds `base_odds`, +`pdo` points per doubling of the odds.
    pdo: float = 20.0
    base_odds: float = 50.0
    base_points: float = 600.0
    max_n_prebins: int = 20
    min_bin_size: float = 0.05
    monotonic_trend: str = "auto_asc_desc"  # bins may only rise or fall monotonically (or one peak/valley)
    min_iv: float = 0.02  # variables with lower information value are dropped
    C: float = 1.0
    special_codes: Mapping[str, Sequence[float]] = field(default_factory=dict)


def build_scorecard(features: Sequence[str], cfg: ScorecardConfig) -> Scorecard:
    fit_params: dict[str, dict[str, Any]] = {f: {"monotonic_trend": cfg.monotonic_trend} for f in features}
    for var, codes in cfg.special_codes.items():
        fit_params.setdefault(var, {})["special_codes"] = list(codes)
    bp = BinningProcess(
        variable_names=list(features),
        max_n_prebins=cfg.max_n_prebins,
        min_bin_size=cfg.min_bin_size,
        selection_criteria={"iv": {"min": cfg.min_iv}},
        binning_fit_params=fit_params,
    )
    return Scorecard(
        binning_process=bp,
        estimator=LogisticRegression(C=cfg.C, max_iter=1000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": cfg.pdo, "odds": cfg.base_odds, "scorecard_points": cfg.base_points},
    )


def fit_predict_pd(
    features: Sequence[str],
    cfg: ScorecardConfig,
    X_train: pd.DataFrame,
    y_train: npt.ArrayLike,
    X_test: pd.DataFrame,
) -> tuple[Scorecard, npt.NDArray[np.float64]]:
    sc = build_scorecard(features, cfg)
    sc.fit(X_train[list(features)], np.asarray(y_train))
    p = np.asarray(sc.predict_proba(X_test[list(features)])[:, 1], dtype=np.float64)
    return sc, p
