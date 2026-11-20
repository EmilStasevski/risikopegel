"""Smoke test (first-session checklist step 4): WoE scorecard on Give Me Some Credit, end to end.

GMSC has no time column, so a stratified holdout is used. This is the documented exception in
docs/split_policy.md: the run supports no stability or drift claims.
Writes results/credit/gmsc_smoke/*.  Run: uv run python scripts/smoke_gmsc_scorecard.py
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optbinning
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from risikopegel.credit.data import GMSC_FEATURES, GMSC_SPECIAL_CODES, GMSC_TARGET, load_gmsc_train
from risikopegel.credit.metrics import calibration_table, pd_metrics
from risikopegel.credit.scorecard import ScorecardConfig, fit_predict_pd
from risikopegel.leakage import shuffled_label_check
from risikopegel.splits import Split

SEED = 20261120
OUT = Path("results/credit/gmsc_smoke")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_gmsc_train()
    X, y = df[GMSC_FEATURES], df[GMSC_TARGET].to_numpy()
    cfg = ScorecardConfig(special_codes=GMSC_SPECIAL_CODES)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=SEED)
    t0 = time.perf_counter()
    sc, p_te = fit_predict_pd(GMSC_FEATURES, cfg, X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te])
    fit_s = time.perf_counter() - t0
    p_tr = sc.predict_proba(X.iloc[idx_tr])[:, 1]

    # 5-fold CV on the training part: spread of the holdout estimate
    cv: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(skf.split(idx_tr, y[idx_tr])):
        ia, ib = idx_tr[a], idx_tr[b]
        _, pb = fit_predict_pd(GMSC_FEATURES, cfg, X.iloc[ia], y[ia], X.iloc[ib])
        cv.append({"fold": k, **pd_metrics(y[ib], pb)})

    # leakage guard: whole pipeline on permuted training labels must drop to AUC ~0.5.
    # IV filter off (min_iv=0): otherwise no variable survives on noise and the check would be vacuous.
    cfg_shuf = ScorecardConfig(special_codes=GMSC_SPECIAL_CODES, min_iv=0.0)
    split = Split(np.sort(idx_tr), np.sort(idx_te), None, None, None, None)

    def fit_score(Xa: pd.DataFrame, ya: Any, Xb: pd.DataFrame, yb: Any) -> float:
        _, pb = fit_predict_pd(GMSC_FEATURES, cfg_shuf, Xa, ya, Xb)
        return float(roc_auc_score(yb, pb))

    shuf = shuffled_label_check(fit_score, X, y, split, chance=0.5, tolerance=0.02, n_rounds=20, seed=SEED)

    bp = sc.binning_process_
    selected = list(bp.get_support(names=True))
    summary = bp.summary()
    coefs = pd.DataFrame({"variable": selected, "coef": sc.estimator_.coef_[0]})

    result = {
        "run": {
            "script": "scripts/smoke_gmsc_scorecard.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": SEED,
            "python": platform.python_version(),
            "optbinning": optbinning.__version__,
            "sklearn": sklearn.__version__,
            "fit_seconds": fit_s,
        },
        "data": {
            "dataset": "Give Me Some Credit (Kaggle), cs-training.csv",
            "rows": len(df),
            "train_rows": len(idx_tr),
            "test_rows": len(idx_te),
            "split": "stratified random 70/30 (no time column; documented exception)",
        },
        "config": {k: v for k, v in cfg.__dict__.items() if k != "special_codes"}
        | {"special_codes": {k: list(v) for k, v in cfg.special_codes.items()}},
        "selected_variables": selected,
        "holdout": pd_metrics(y[idx_te], p_te),
        "train": pd_metrics(y[idx_tr], p_tr),
        "cv5_on_train": cv,
        "shuffled_label": {
            "auc_scores": shuf.scores.tolist(),
            "mean": shuf.mean,
            "sd": shuf.sd,
            "se": shuf.se,
            "z": shuf.z,
            "chance": shuf.chance,
            "tolerance": shuf.tolerance,
            "passed": shuf.passed,
            "note": "IV selection disabled (min_iv=0) so binning + regression still fit on noise",
        },
        "public_reference": {
            "source": "Kaggle CLI `kaggle competitions leaderboard -c GiveMeSomeCredit --show`, "
            "fetched 2026-11-20",
            "top_leaderboard_auc": 0.86955,
            "comparable": False,
            "note": "Leaderboard AUC is on the unlabelled Kaggle test set; "
            "this holdout is a different sample.",
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    summary.to_csv(OUT / "binning_summary.csv", index=False)
    coefs.to_csv(OUT / "coefficients.csv", index=False)
    calibration_table(y[idx_te], p_te).to_csv(OUT / "calibration_holdout.csv", index=False)
    pd.concat([bp.get_binned_variable(v).binning_table.build().assign(variable=v) for v in selected]).to_csv(
        OUT / "bin_tables.csv", index=False
    )
    sc.table(style="summary").to_csv(OUT / "scorecard_points.csv", index=False)
    print(json.dumps(result["holdout"], indent=2), "\nshuffled-label passed:", shuf.passed)


if __name__ == "__main__":
    main()
