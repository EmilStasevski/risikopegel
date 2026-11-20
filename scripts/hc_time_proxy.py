"""Is there a usable time proxy in Home Credit? (R1 task; the data has only relative day offsets.)

Checks, all on mart_hc__applicant_features (dbt):
1. SK_ID_CURR vs default: Spearman correlation and default rate per ID decile (chi-square test).
2. Adversarial validation by ID: can features tell low IDs from high IDs? AUC ~0.5 => IDs carry no
   time/drift signal that the features can see.
3. Adversarial validation train vs Kaggle test: AUC ~0.5 => the Kaggle test set is a random draw,
   not a later period.
Writes results/credit/hc_time_proxy.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

DB = Path("data/processed/risikopegel.duckdb")
OUT = Path("results/credit/hc_time_proxy.json")
SEED = 20261120
NON_FEATURES = {"sk_id_curr", "is_train", "target"}


def adversarial_auc(X: pd.DataFrame, label: np.ndarray[Any, Any]) -> dict[str, Any]:
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p = cross_val_predict(clf, X, label, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(label, p))
    # top drivers via a single fit's split importance proxy: univariate AUC per column
    uni = {}
    for c in X.columns:
        col = X[c]
        if col.notna().sum() < 1000 or col.nunique() < 2:
            continue
        m = col.notna().to_numpy()
        a = roc_auc_score(label[m], col[m].astype(float))
        uni[c] = abs(a - 0.5)
    top = sorted(uni.items(), key=lambda kv: -kv[1])[:5]
    return {"cv5_auc": auc, "top_univariate_abs_auc_minus_half": {k: round(v, 4) for k, v in top}}


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("select * from mart_hc__applicant_features").df()
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in {"string", "str"}:
            codes = X[c].astype("category").cat.codes.astype(float)
            X[c] = codes.where(codes >= 0)  # -1 = missing category
        elif X[c].dtype == bool:
            X[c] = X[c].astype(float)
    X = X.astype(float)

    tr = df["is_train"].to_numpy()
    ids, y = df.loc[tr, "sk_id_curr"].to_numpy(), df.loc[tr, "target"].to_numpy().astype(int)
    rho, rho_p = stats.spearmanr(ids, y)
    dec = pd.qcut(ids, 10, labels=False)
    table = pd.crosstab(dec, y)
    chi2, chi_p, _, _ = stats.chi2_contingency(table)
    rate_by_decile = (table[1] / table.sum(axis=1)).round(5).tolist()

    high_id = (ids > np.median(ids)).astype(int)
    adv_id = adversarial_auc(X.loc[tr].reset_index(drop=True), high_id)
    adv_tt = adversarial_auc(X, tr.astype(int))
    # Ablations: bureau_balance coverage differs sharply between train and Kaggle test (data artefact),
    # and credit/annuity is the strongest single separator. How much shift remains without them?
    bb = ["bureau_worst_dpd_bucket", "bureau_worst_dpd_bucket_12m"]
    no_bb = [c for c in X.columns if c not in bb]
    no_bb_term = [c for c in no_bb if c != "credit_term_years"]
    adv_tt_no_bb = adversarial_auc(X[no_bb], tr.astype(int))
    adv_tt_no_bb_term = adversarial_auc(X[no_bb_term], tr.astype(int))
    has_bureau = df["bureau_n_loans"] > 0
    bb_cov = {
        ("train" if k else "kaggle_test"): float(
            df.loc[has_bureau & (df["is_train"] == k), bb[0]].notna().mean()
        )
        for k in (True, False)
    }

    result = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows_train": int(tr.sum()),
        "rows_test": int((~tr).sum()),
        "n_features": len(feats),
        "id_vs_default": {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "chi2_default_rate_by_id_decile": float(chi2),
            "chi2_p": float(chi_p),
            "default_rate_by_id_decile": rate_by_decile,
        },
        "adversarial_low_vs_high_id": adv_id,
        "adversarial_train_vs_kaggle_test": adv_tt,
        "adversarial_train_vs_kaggle_test_without_bureau_balance": adv_tt_no_bb,
        "adversarial_train_vs_kaggle_test_without_bureau_balance_and_credit_annuity": adv_tt_no_bb_term,
        "bureau_balance_coverage_given_bureau_record": bb_cov,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
