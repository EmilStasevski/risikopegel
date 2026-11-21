"""R2: hyperparameter search for every model family (scorecard included), Optuna TPE.

Protocol (same discipline as the grid run):
- fit on train, early stopping + selection on valid AUC only; test is never touched here
- the sampler is seeded and the first trial of each family is the grid winner (warm start), so the
  search can only improve on the grid on valid
- fixed budget per family (n_trials, wall-clock cap); every trial is logged
- the selected params feed `scripts/r2_challengers.py --params`, which scores test once and compares
  against the grid baseline (paired bootstrap)

Selecting the best of many trials on valid makes the best valid AUC optimistic; the valid-test gap is
reported for exactly that reason.

Writes results/credit/hc_tuning/{best_params.json, trials.csv}; the Optuna study (resumable) goes to
data/processed/hc_tuning/optuna.db.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import SEED, fit_family

from risikopegel.credit.challengers import monotone_signs
from risikopegel.credit.hc import load_mart, prepare

OUT = Path("results/credit/hc_tuning")
DB = Path("data/processed/hc_tuning/optuna.db")
GRID_RESULTS = Path("results/credit/hc_challengers_grid/results.json")
log = logging.getLogger("r2_tune")

# (n_trials, timeout seconds). Budgets sized from the grid run's fit times (EBM ~1.5 min, CatBoost ~1 min).
BUDGET: dict[str, tuple[int, int]] = {
    "scorecard": (30, 1800),
    "lightgbm": (60, 2700),
    "lightgbm_monotone": (30, 2700),
    "xgboost": (60, 2700),
    "catboost": (40, 3600),
    "ebm": (20, 3600),
}


# Fitter defaults for the searched params (credit/challengers.py, library defaults), so that the
# warm-start trial reproduces the grid winner. reg_alpha=0 is outside the log range; 1e-3 is equivalent.
DEFAULTS: dict[str, dict[str, Any]] = {
    "scorecard": {"C": 1.0, "min_iv": 0.02, "max_n_prebins": 20, "min_bin_size": 0.05},
    "lightgbm": {"learning_rate": 0.05, "reg_alpha": 1e-3, "colsample_bytree": 0.7, "subsample": 0.8},
    "xgboost": {
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "reg_alpha": 1e-3,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
    "catboost": {
        "learning_rate": 0.05,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "border_count": 254,
    },
    "ebm": {
        "learning_rate": 0.015,
        "max_bins": 1024,
        "min_samples_leaf": 4,
        "max_leaves": 2,
        "smoothing_rounds": 75,
    },
}
DEFAULTS["lightgbm_monotone"] = DEFAULTS["lightgbm"]


def _lgb(t: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": t.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 1000, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
    }


SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "scorecard": lambda t: {
        "C": t.suggest_float("C", 1e-2, 10.0, log=True),
        "min_iv": t.suggest_float("min_iv", 0.0, 0.05),
        "max_n_prebins": t.suggest_int("max_n_prebins", 10, 40),
        "min_bin_size": t.suggest_float("min_bin_size", 0.02, 0.08),
    },
    "lightgbm": _lgb,
    "lightgbm_monotone": _lgb,
    "xgboost": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": t.suggest_float("gamma", 0.0, 5.0),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
    },
    "catboost": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "depth": t.suggest_int("depth", 4, 9),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": t.suggest_int("border_count", 64, 254),
    },
    "ebm": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "interactions": t.suggest_int("interactions", 0, 30),
        "max_bins": t.suggest_categorical("max_bins", [256, 512, 1024]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 2, 50, log=True),
        "max_leaves": t.suggest_int("max_leaves", 2, 3),
        "smoothing_rounds": t.suggest_int("smoothing_rounds", 0, 200),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(BUDGET))
    ap.add_argument("--summarize", action="store_true", help="only rewrite the outputs from the stored study")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    d = prepare(load_mart(), seed=SEED)
    signs = monotone_signs(d.X_train, d.y_train, d.numeric)
    grid = json.loads(GRID_RESULTS.read_text())["models"]

    for name in [] if args.summarize else args.families:
        n_trials, timeout = BUDGET[name]
        study = optuna.create_study(
            study_name=f"hc_{name}",
            storage=f"sqlite:///{DB}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=10),
            load_if_exists=True,
        )
        if len(study.trials) == 0:
            study.enqueue_trial({**DEFAULTS[name], **grid[name]["selected_params"]}, skip_if_exists=True)
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        def objective(t: optuna.Trial, name: str = name) -> float:
            params = SPACES[name](t)
            f = fit_family(name, params, d, signs)
            auc = float(roc_auc_score(d.y_valid, f.predict(d.X_valid)))
            t.set_user_attr("params", params)
            t.set_user_attr("fit_seconds", f.fit_seconds)
            t.set_user_attr("best_iteration", f.best_iteration)
            log.info("%-18s #%d valid_auc=%.5f fit=%.0fs it=%s", name, t.number, auc, f.fit_seconds,
                     f.best_iteration)  # fmt: skip
            return auc

        if done < n_trials:
            study.optimize(objective, n_trials=n_trials - done, timeout=timeout)

    rows, best_params, summary = [], {}, {}
    for name in BUDGET:
        try:
            study = optuna.load_study(study_name=f"hc_{name}", storage=f"sqlite:///{DB}")
        except KeyError:
            continue
        ok = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in ok:
            rows.append({"family": name, "trial": t.number, "valid_auc": t.value,
                         "fit_seconds": t.user_attrs.get("fit_seconds"),
                         "best_iteration": t.user_attrs.get("best_iteration"),
                         "params": json.dumps(t.user_attrs.get("params", t.params))})  # fmt: skip
        b = max(ok, key=lambda t: float(t.value or 0.0))
        # Trial 0 is the grid winner up to reg_alpha 0 -> 1e-3 and thread-level nondeterminism, which can move
        # the early-stopping point. Guarantee "never worse than the grid on valid" explicitly.
        kept_grid = float(b.value or 0.0) <= grid[name]["valid_auc"]
        best_params[name] = (
            grid[name]["selected_params"] if kept_grid else b.user_attrs.get("params", b.params)
        )
        summary[name] = {
            "n_trials": len(ok),
            "best_trial": None if kept_grid else b.number,
            "kept_grid_params": kept_grid,
            "trial0_valid_auc": next(float(t.value or 0.0) for t in ok if t.number == 0),
            "best_valid_auc": b.value,
            "grid_valid_auc": grid[name]["valid_auc"],
            "search_seconds": float(sum(t.user_attrs.get("fit_seconds", 0.0) for t in ok)),
        }
    pd.DataFrame(rows).to_csv(OUT / "trials.csv", index=False)
    (OUT / "best_params.json").write_text(json.dumps({
        "run": {"script": "scripts/r2_tune.py", "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "seed": SEED, "optuna": optuna.__version__, "sampler": "TPE multivariate, 10 startup",
                "budget": BUDGET, "warm_start": "trial 0 = grid winner; grid kept unless beaten"},
        "summary": summary,
        "best_params": best_params,
    }, indent=2))  # fmt: skip
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
