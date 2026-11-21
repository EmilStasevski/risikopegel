"""R2: WoE scorecard vs boosting challengers on Home Credit (Q1 / hypothesis H1).

Protocol (pre-set, see docs/split_policy.md and docs/hypotheses.md):
- stratified 64/16/20 train/valid/test (no time proxy exists); seed fixed
- each model family: small fixed grid (GRIDS) or, with --params, the Optuna-selected params from
  scripts/r2_tune.py; early stopping and selection on valid AUC only
- test is scored once per family with the selected config
- paired bootstrap (1000 resamples of test) for AUC differences vs the scorecard
- calibration raw and after Platt scaling fitted on valid
- stability: score PSI train -> Kaggle test (drift scenario) and valid -> test (sanity, should be ~0)

Writes results/credit/hc_challengers/{tuning.json, results.json, csi.csv};
models + test predictions -> data/processed/hc_challengers/ (git-ignored).
"""

from __future__ import annotations

import argparse
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
from sklearn.isotonic import IsotonicRegression
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
APPROVAL_RATE = 0.85  # illustrative policy for the decision view; not a recommendation
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


def fit_family(name: str, params: dict[str, Any], d: HCData, signs: dict[str, int]) -> Fitted:
    a = (params, d.X_train, d.y_train, d.X_valid, d.y_valid)
    if name == "scorecard":
        return fit_scorecard(*a, seed=SEED, categorical=d.categorical)
    if name == "lightgbm":
        return fit_lightgbm(*a, seed=SEED)
    if name == "lightgbm_monotone":
        return fit_lightgbm(*a, seed=SEED, monotone=signs)
    if name == "catboost":
        return fit_catboost(*a, seed=SEED, categorical=d.categorical)
    if name == "xgboost":
        return fit_xgboost(*a, seed=SEED)
    if name == "ebm":
        return fit_ebm(*a, seed=SEED)
    raise ValueError(name)


def run_family(
    name: str, d: HCData, signs: dict[str, int], grids: dict[str, dict[str, list[Any]]]
) -> tuple[Fitted, list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    best: Fitted | None = None
    best_auc = -1.0
    for params in grid(grids[name]):
        f = fit_family(name, params, d, signs)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, help="best_params.json from scripts/r2_tune.py (replaces GRIDS)")
    args = ap.parse_args()
    grids, search = GRIDS, "grid"
    if args.params:
        best = json.loads(args.params.read_text())["best_params"]
        grids = {k: {p: [v] for p, v in best[k].items()} for k in GRIDS}
        search = f"optuna (TPE), selected params from {args.params}"
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
    for name in grids:
        fitted[name], tuning[name] = run_family(name, d, signs, grids)
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
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p_va, d.y_valid)
        p_te_iso = np.asarray(iso.predict(p_te), dtype=np.float64)
        preds[name] = {"test": p_te, "test_platt": p_te_cal, "train": p_tr, "valid": p_va, "kaggle": p_kg}
        res[name] = {
            "selected_params": f.params,
            "best_iteration": f.best_iteration,
            "fit_seconds": f.fit_seconds,
            "predict_seconds_test": pred_s,
            "extra": f.extra,
            "test": pd_metrics(d.y_test, p_te),
            "test_platt": pd_metrics(d.y_test, p_te_cal),
            "test_isotonic": pd_metrics(d.y_test, p_te_iso),
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
        # decision view: cutoff fixed on VALID so that APPROVAL_RATE of applicants are approved
        cut = float(np.quantile(preds[name]["valid"], APPROVAL_RATE))
        approve = preds[name]["test"] <= cut
        res[name]["at_cutoff"] = {"approval_rate_target": APPROVAL_RATE, "pd_cutoff": cut, "groups": {}}
        for k in ("all", "F", "M"):
            m = np.ones_like(approve) if k == "all" else (g == k)
            yk, ak = d.y_test[m], approve[m]
            res[name]["at_cutoff"]["groups"][k] = {
                "approval_rate": float(ak.mean()),
                "default_rate_approved": float(yk[ak].mean()),
                "share_defaulters_rejected": float((~ak[yk == 1]).mean()),
                "share_goods_rejected": float((~ak[yk == 0]).mean()),
            }

    csi = csi_table(pd.concat([d.X_train]), d.X_kaggle)
    csi.to_csv(OUT / "csi_train_to_kaggle.csv", index=False)

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
        "search": search,
        "grids": grids,
        "models": res,
        # champion chosen on VALID AUC only (never on test), among the challengers
        "champion": {
            "name": max((k for k in res if k != "scorecard"), key=lambda k: res[k]["valid_auc"]),
            "rule": "highest valid AUC among challengers",
        },
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float))
    # artifacts after results: a failure here must not lose the metrics
    for part, yv in (("test", d.y_test), ("valid", d.y_valid)):
        pd.DataFrame({k: v[part] for k, v in preds.items()}).assign(y=yv).to_parquet(
            ART / f"{part}_predictions.parquet"
        )
    pd.DataFrame({k: v["kaggle"] for k, v in preds.items()}).to_parquet(ART / "kaggle_predictions.parquet")
    for name, r in res.items():
        log.info("%-18s test AUC %.4f  Brier %.5f  slope %.3f  PSI(kaggle) %.3f", name, r["test"]["auc"],
                 r["test"]["brier"], r["test"]["calibration_slope"], r["psi_train_to_kaggle"])  # fmt: skip


if __name__ == "__main__":
    main()
