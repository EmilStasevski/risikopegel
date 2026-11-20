"""R2: WoE scorecard vs boosting challengers on Home Credit (Q1 / hypothesis H1).

Protocol (pre-set, see docs/split_policy.md and docs/hypotheses.md):
- stratified 64/16/20 train/valid/test (no time proxy exists); seed fixed
- each model family: small fixed grid, early stopping and selection on valid AUC only
- test is scored once per family with the selected config
- paired bootstrap (1000 resamples of test) for AUC differences vs the scorecard
- calibration raw and after Platt scaling fitted on valid
- stability: score PSI train -> Kaggle test (drift scenario) and valid -> test (sanity, should be ~0)

Writes results/credit/hc_challengers/{tuning.json, results.json, csi.csv};
models + test predictions -> data/processed/hc_challengers/ (git-ignored).
"""

from __future__ import annotations

import json
import logging
import pickle
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import catboost
import interpret
import lightgbm
import numpy as np
import numpy.typing as npt
import optbinning
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from risikopegel.credit.challengers import (
    Fitted,
    fit_catboost,
    fit_ebm,
    fit_lightgbm,
    fit_scorecard,
    fit_xgboost,
    grid,
    monotone_signs,
)
from risikopegel.credit.hc import HCData, load_mart, prepare
from risikopegel.credit.metrics import calibration_table, pd_metrics
from risikopegel.credit.monitoring import csi_table, psi

SEED = 20261120
OUT = Path("results/credit/hc_challengers")
ART = Path("data/processed/hc_challengers")
N_BOOT = 1000
log = logging.getLogger("r2")

GRIDS: dict[str, dict[str, list[Any]]] = {
    "scorecard": {"C": [0.1, 1.0], "min_iv": [0.02]},
    "lightgbm": {"num_leaves": [15, 31, 63], "min_child_samples": [50, 200], "reg_lambda": [0.0, 5.0]},
    "lightgbm_monotone": {"num_leaves": [15, 31, 63], "min_child_samples": [200], "reg_lambda": [5.0]},
    "catboost": {"depth": [4, 6, 8], "l2_leaf_reg": [3.0, 10.0]},
    "xgboost": {"max_depth": [3, 5, 7], "min_child_weight": [1.0, 10.0]},
    "ebm": {"interactions": [0, 10]},
}


def fast_auc(y: npt.NDArray[Any], p: npt.NDArray[Any]) -> float:
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_bootstrap(
    y: npt.NDArray[Any], p_ref: npt.NDArray[Any], p_new: npt.NDArray[Any], rng: np.random.Generator
) -> dict[str, float]:
    diffs = np.empty(N_BOOT)
    n = y.size
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs[b] = fast_auc(y[i], p_new[i]) - fast_auc(y[i], p_ref[i])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "auc_diff": fast_auc(y, p_new) - fast_auc(y, p_ref),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "share_diff_le_0": float((diffs <= 0).mean()),
    }


def platt(
    p_valid: npt.NDArray[Any], y_valid: npt.NDArray[Any]
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    def logit(p: npt.NDArray[Any]) -> npt.NDArray[Any]:
        q = np.clip(p, 1e-9, 1 - 1e-9)
        out: npt.NDArray[Any] = np.log(q / (1 - q)).reshape(-1, 1)
        return out

    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(p_valid), y_valid)
    return lambda p: np.asarray(lr.predict_proba(logit(p))[:, 1], dtype=np.float64)


def run_family(name: str, d: HCData, signs: dict[str, int]) -> tuple[Fitted, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    best: Fitted | None = None
    best_auc = -1.0
    for params in grid(GRIDS[name]):
        if name == "scorecard":
            f = fit_scorecard(
                params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED, categorical=d.categorical
            )
        elif name == "lightgbm":
            f = fit_lightgbm(params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED)
        elif name == "lightgbm_monotone":
            f = fit_lightgbm(params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED, monotone=signs)
        elif name == "catboost":
            f = fit_catboost(
                params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED, categorical=d.categorical
            )
        elif name == "xgboost":
            f = fit_xgboost(params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED)
        elif name == "ebm":
            f = fit_ebm(params, d.X_train, d.y_train, d.X_valid, d.y_valid, seed=SEED)
        else:
            raise ValueError(name)
        auc_va = float(roc_auc_score(d.y_valid, f.predict(d.X_valid)))
        trials.append(
            {
                "params": params,
                "valid_auc": auc_va,
                "fit_seconds": f.fit_seconds,
                "best_iteration": f.best_iteration,
            }
        )
        log.info(
            "%-18s %s valid_auc=%.5f fit=%.1fs it=%s", name, params, auc_va, f.fit_seconds, f.best_iteration
        )
        if auc_va > best_auc:
            best, best_auc = f, auc_va
    assert best is not None
    return best, trials


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    d = prepare(load_mart(), seed=SEED)
    log.info("train %d valid %d test %d kaggle %d features %d (numeric %d, categorical %d)",
             len(d.y_train), len(d.y_valid), len(d.y_test), len(d.X_kaggle),
             d.X_train.shape[1], len(d.numeric), len(d.categorical))  # fmt: skip
    signs = monotone_signs(d.X_train, d.y_train, d.numeric)

    fitted: dict[str, Fitted] = {}
    tuning: dict[str, Any] = {}
    for name in GRIDS:
        fitted[name], tuning[name] = run_family(name, d, signs)
    (OUT / "tuning.json").write_text(json.dumps(tuning, indent=2))

    rng = np.random.default_rng(SEED)
    preds: dict[str, dict[str, npt.NDArray[Any]]] = {}
    res: dict[str, Any] = {}
    for name, f in fitted.items():
        t0 = time.perf_counter()
        p_te = f.predict(d.X_test)
        pred_s = time.perf_counter() - t0
        p_va, p_tr, p_kg = f.predict(d.X_valid), f.predict(d.X_train), f.predict(d.X_kaggle)
        cal = platt(p_va, d.y_valid)
        p_te_cal = cal(p_te)
        preds[name] = {"test": p_te, "test_platt": p_te_cal, "train": p_tr, "valid": p_va, "kaggle": p_kg}
        res[name] = {
            "selected_params": f.params,
            "best_iteration": f.best_iteration,
            "fit_seconds": f.fit_seconds,
            "predict_seconds_test": pred_s,
            "extra": f.extra,
            "test": pd_metrics(d.y_test, p_te),
            "test_platt": pd_metrics(d.y_test, p_te_cal),
            "train_auc": float(roc_auc_score(d.y_train, p_tr)),
            "valid_auc": float(roc_auc_score(d.y_valid, p_va)),
            "psi_train_to_kaggle": psi(p_tr, p_kg),
            "psi_valid_to_test": psi(p_va, p_te),
            "calibration_test": calibration_table(d.y_test, p_te).to_dict(orient="list"),
            "calibration_test_platt": calibration_table(d.y_test, p_te_cal).to_dict(orient="list"),
        }
        with open(ART / f"{name}.pkl", "wb") as fh:
            pickle.dump(f.model, fh)

    for name in fitted:
        if name != "scorecard":
            res[name]["vs_scorecard"] = paired_bootstrap(
                d.y_test, preds["scorecard"]["test"], preds[name]["test"], rng
            )
    # basic group check on a protected-like attribute (not used as a feature)
    g = d.protected_test["gender"].astype(str).to_numpy()
    for name in fitted:
        res[name]["by_gender"] = {
            k: {
                "n": int((g == k).sum()),
                "auc": float(roc_auc_score(d.y_test[g == k], preds[name]["test"][g == k])),
                "default_rate": float(d.y_test[g == k].mean()),
                "mean_pd": float(preds[name]["test"][g == k].mean()),
            }
            for k in ("F", "M")
        }

    csi = csi_table(pd.concat([d.X_train]), d.X_kaggle)
    csi.to_csv(OUT / "csi_train_to_kaggle.csv", index=False)
    pd.DataFrame({k: v["test"] for k, v in preds.items()}).assign(y=d.y_test).to_parquet(
        ART / "test_predictions.parquet"
    )

    out = {
        "run": {
            "script": "scripts/r2_challengers.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": SEED,
            "python": platform.python_version(),
            "versions": {
                "lightgbm": lightgbm.__version__,
                "catboost": catboost.__version__,
                "xgboost": xgboost.__version__,
                "interpret": interpret.__version__,
                "optbinning": optbinning.__version__,
                "sklearn": sklearn.__version__,
            },
            "n_bootstrap": N_BOOT,
        },
        "data": {
            "dataset": "Home Credit Default Risk (Kaggle), dbt mart_hc__applicant_features",
            "split": "stratified 64/16/20 train/valid/test (no time proxy; documented exception)",
            "train": len(d.y_train),
            "valid": len(d.y_valid),
            "test": len(d.y_test),
            "kaggle_unlabelled": len(d.X_kaggle),
            "n_features": d.X_train.shape[1],
            "categorical": d.categorical,
            "excluded_from_features": ["gender (fairness check only)"],
            "test_default_rate": float(d.y_test.mean()),
        },
        "monotone_signs": signs,
        "grids": GRIDS,
        "models": res,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float))
    for name, r in res.items():
        log.info("%-18s test AUC %.4f  Brier %.5f  slope %.3f  PSI(kaggle) %.3f", name, r["test"]["auc"],
                 r["test"]["brier"], r["test"]["calibration_slope"], r["psi_train_to_kaggle"])  # fmt: skip


if __name__ == "__main__":
    main()
