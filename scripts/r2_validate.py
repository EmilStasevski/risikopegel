"""R2: independent validation checks for the Home Credit PD champion (and the scorecard as benchmark).

Written from the validator's side: it only reads the developer's artifacts (models, predictions,
results.json) and the shared data mart, and re-derives everything it reports. Checks:

1. Reproducibility: reload the pickled models, re-score test, compare with the stored predictions.
2. Rating-grade calibration: 10 grades with PD boundaries fixed on valid; per grade on test the
   one-sided Jeffreys test (H0: grade PD >= true default rate; small p = PD too low), as in the ECB's
   2019 validation-reporting instructions for IRB PD back-testing; rank ordering of default rates.
3. Split-seed stability: re-split with other seeds, refit champion and scorecard with the selected
   params, re-score test. Separates "the gain is real" from "the gain is one lucky split".
4. Segment performance: AUC by contract type, bureau-history status and income type.
5. Sensitivity: the external credit scores (ext_source_1..3) are the top drivers. Scenario: they are
   unavailable (set to missing) for all test applicants; how much discrimination is left?

Writes results/credit/hc_validation/validation.json.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import ART, OUT, SEED, fit_family

from risikopegel.credit.challengers import monotone_signs
from risikopegel.credit.hc import load_mart, prepare

VAL = Path("results/credit/hc_validation")
N_GRADES = 10
ALPHA = 0.05
SPLIT_SEEDS = [11, 22, 33]
SEGMENTS = ["contract_type", "bureau_history_status", "income_type"]
MIN_SEG_N, MIN_SEG_D = 1000, 50
EXT = ["ext_source_1", "ext_source_2", "ext_source_3"]


def jeffreys_p(n: int, d: int, pd_: float) -> float:
    """One-sided Jeffreys p-value: posterior Beta(d+1/2, n-d+1/2) mass below the grade PD."""
    return float(stats.beta.cdf(pd_, d + 0.5, n - d + 0.5))


def grade_table(p_valid: npt.NDArray[Any], p_test: npt.NDArray[Any], y: npt.NDArray[Any]) -> dict[str, Any]:
    edges = np.quantile(p_valid, np.linspace(0, 1, N_GRADES + 1)[1:-1])
    g = np.searchsorted(edges, p_test, side="right")  # grade 0 = lowest PD
    rows = []
    for k in range(N_GRADES):
        m = g == k
        n, d = int(m.sum()), int(y[m].sum())
        mean_pd = float(p_test[m].mean())
        rows.append({"grade": k + 1, "n": n, "defaults": d, "mean_pd": mean_pd, "default_rate": d / n,
                     "jeffreys_p": jeffreys_p(n, d, mean_pd)})  # fmt: skip
    t = pd.DataFrame(rows)
    dr = t["default_rate"].to_numpy()
    return {
        "grades": rows,
        "n_rejected_underestimation": int((t["jeffreys_p"] < ALPHA).sum()),
        # overestimation (PD too conservative): mirror test, reported for completeness
        "n_rejected_overestimation": int((1 - t["jeffreys_p"] < ALPHA).sum()),
        "default_rate_monotone": bool(np.all(np.diff(dr) > 0)),
        "spearman_grade_vs_default_rate": float(stats.spearmanr(t["grade"], dr).statistic),
    }


def split_seed_stability(df: pd.DataFrame, res: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Re-split with other seeds, refit with the selected params, re-score test (slowest check)."""
    champ = names[0]
    per_seed = []
    for s in SPLIT_SEEDS:
        ds = prepare(df, seed=s)
        signs = monotone_signs(ds.X_train, ds.y_train, ds.numeric)
        row: dict[str, Any] = {"split_seed": s}
        for k in names:
            f = fit_family(k, res["models"][k]["selected_params"], ds, signs)
            row[k] = float(roc_auc_score(ds.y_test, f.predict(ds.X_test)))
        row["delta_auc"] = row[champ] - row["scorecard"]
        per_seed.append(row)
        print(row, flush=True)
    deltas = np.array([r["delta_auc"] for r in per_seed])
    return {
        "per_seed": per_seed,
        "note": "params as selected on the main split (no re-tuning per seed)",
        "delta_auc_mean": float(deltas.mean()),
        "delta_auc_min": float(deltas.min()),
        "delta_auc_max": float(deltas.max()),
        "champion_auc_sd": float(np.std([r[champ] for r in per_seed], ddof=1)),
        "scorecard_auc_sd": float(np.std([r["scorecard"] for r in per_seed], ddof=1)),
    }


def main() -> None:
    VAL.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = json.loads((OUT / "results.json").read_text())
    champ = res["champion"]["name"]
    names = [champ, "scorecard"]
    df = load_mart()
    d = prepare(df, seed=SEED)
    models = {k: pickle.loads((ART / f"{k}.pkl").read_bytes()) for k in names}
    stored = pd.read_parquet(ART / "test_predictions.parquet")
    valid_stored = pd.read_parquet(ART / "valid_predictions.parquet")
    out: dict[str, Any] = {
        "run": {"script": "scripts/r2_validate.py", "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "alpha": ALPHA, "n_grades": N_GRADES, "split_seeds": SPLIT_SEEDS},
        "champion": champ,
        "champion_params": res["models"][champ]["selected_params"],
    }  # fmt: skip

    def predict(name: str, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        m = models[name]
        if name == "scorecard":
            Xo = X.copy()
            for c in d.categorical:
                Xo[c] = Xo[c].astype(str)
            return np.asarray(m.predict_proba(Xo)[:, 1], dtype=np.float64)
        return np.asarray(m.predict_proba(X)[:, 1], dtype=np.float64)

    # 1. reproducibility
    out["reproducibility"] = {}
    rescored: dict[str, npt.NDArray[np.float64]] = {}
    for k in names:
        rescored[k] = predict(k, d.X_test)
        out["reproducibility"][k] = {
            "max_abs_diff_vs_stored": float(np.abs(rescored[k] - stored[k].to_numpy()).max()),
            "labels_identical": bool(np.array_equal(stored["y"].to_numpy(), d.y_test)),
        }

    # 2. rating-grade calibration
    out["grades"] = {k: grade_table(valid_stored[k].to_numpy(), rescored[k], d.y_test) for k in names}
    # 2b. does the isotonic recalibration (fitted on valid, as in the developer's run) fix grade-level bias?
    pv = valid_stored[champ].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(pv, d.y_valid)
    out["grades"][f"{champ}_isotonic"] = grade_table(
        np.asarray(iso.predict(pv), dtype=np.float64),
        np.asarray(iso.predict(rescored[champ]), dtype=np.float64),
        d.y_test,
    )
    out["bonferroni_alpha_per_grade"] = ALPHA / N_GRADES
    for g in out["grades"].values():
        g["n_rejected_underestimation_bonferroni"] = sum(
            r["jeffreys_p"] < ALPHA / N_GRADES for r in g["grades"]
        )

    # 4. segments
    seg_out: dict[str, Any] = {}
    for s in SEGMENTS:
        vals = d.X_test[s].astype(str).to_numpy()
        for v in pd.unique(vals):
            m = vals == v
            n, dd = int(m.sum()), int(d.y_test[m].sum())
            if n < MIN_SEG_N or dd < MIN_SEG_D:
                continue
            a = {k: float(roc_auc_score(d.y_test[m], rescored[k][m])) for k in names}
            seg_out[f"{s}={v}"] = {"n": n, "defaults": dd, "auc": a, "delta_auc": a[champ] - a["scorecard"]}
    out["segments"] = seg_out

    # 5. sensitivity: external scores unavailable
    Xs = d.X_test.copy()
    Xs[EXT] = np.nan
    sens = {}
    for k in names:
        p = predict(k, Xs)
        sens[k] = {
            "auc_base": float(roc_auc_score(d.y_test, rescored[k])),
            "auc_ext_missing": float(roc_auc_score(d.y_test, p)),
            "mean_pd_base": float(rescored[k].mean()),
            "mean_pd_ext_missing": float(p.mean()),
        }
    out["sensitivity_ext_sources_missing"] = sens

    out["split_seed_stability"] = split_seed_stability(df, res, names)
    (VAL / "validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("reproducibility", "sensitivity_ext_sources_missing")}, indent=2))


if __name__ == "__main__":
    main()
