"""R2: explanations and reason codes for the challengers vs the WoE scorecard.

Needs the artifacts of scripts/r2_challengers.py (same seed -> identical split). Writes
results/credit/hc_explain/{explain.json, global_importance.csv}. Per-applicant explanations stay in
data/processed (they are derived from licensed rows and are not committed).
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import ART, OUT, SEED

from risikopegel.credit.explain import Explanation, explain, logit, reason_agreement, reason_codes
from risikopegel.credit.hc import load_mart, prepare

EXP = Path("results/credit/hc_explain")
N_GLOBAL = 10_000  # test rows for the cross-model global-importance comparison


def global_importance(e: Explanation) -> pd.Series:
    return e.contributions.abs().mean().sort_values(ascending=False)


def per_feature(imp: pd.Series) -> pd.Series:
    """Fold EBM pair terms ('a & b') into their features, half each, for cross-model comparison."""
    out: dict[str, float] = {}
    for term, v in imp.items():
        parts = [t.strip() for t in str(term).split(" & ")]
        for p in parts:
            out[p] = out.get(p, 0.0) + float(v) / len(parts)
    return pd.Series(out).sort_values(ascending=False)


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = json.loads((OUT / "results.json").read_text())
    champion = res["champion"]["name"]
    d = prepare(load_mart(), seed=SEED)
    models = {k: pickle.loads((ART / f"{k}.pkl").read_bytes()) for k in res["models"]}
    preds = pd.read_parquet(ART / "test_predictions.parquet")

    out: dict[str, Any] = {
        "run": {"script": "scripts/r2_explain.py", "utc": datetime.now(UTC).isoformat(timespec="seconds")},
        "champion": champion,
        "models": {},
    }
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(len(d.X_test), N_GLOBAL, replace=False))
    imps: dict[str, pd.Series] = {}
    full: dict[str, Explanation] = {}
    for name, m in models.items():
        X = d.X_test if name in (champion, "scorecard") else d.X_test.iloc[sample].reset_index(drop=True)
        e = explain(name, m, X, d.categorical)
        p = preds[name].to_numpy() if len(X) == len(d.X_test) else preds[name].to_numpy()[sample]
        err = np.abs(e.reconstructed_logit() - logit(p))
        imp = global_importance(e)
        imps[name] = per_feature(imp)
        out["models"][name] = {
            "method": e.method,
            "rows_explained": len(X),
            "additivity_max_abs_error_logit": float(err.max()),
            "additivity_p99_abs_error_logit": float(np.quantile(err, 0.99)),
            "top10_terms": {str(k): float(v) for k, v in imp.head(10).items()},
            "n_terms": int(e.contributions.shape[1]),
        }
        if name in (champion, "scorecard"):
            full[name] = e

    # cross-model agreement of global feature rankings (Spearman over the union of features)
    feats = sorted(set().union(*(set(s.index) for s in imps.values())))
    mat = pd.DataFrame({k: v.reindex(feats).fillna(0.0) for k, v in imps.items()})
    mat.to_csv(EXP / "global_importance.csv", index_label="feature")
    out["importance_rank_corr_vs_champion"] = {
        k: float(stats.spearmanr(mat[champion], mat[k]).statistic) for k in mat.columns if k != champion
    }

    # reason codes for applicants rejected by BOTH models at the validation-fixed cutoff
    cut_c = res["models"][champion]["at_cutoff"]["pd_cutoff"]
    cut_s = res["models"]["scorecard"]["at_cutoff"]["pd_cutoff"]
    rej_c = preds[champion].to_numpy() > cut_c
    rej_s = preds["scorecard"].to_numpy() > cut_s
    both = np.flatnonzero(rej_c & rej_s)
    sc_contrib = full["scorecard"].contributions
    # scorecard convention: points lost vs the best bin of each variable (best = lowest log-odds contribution)
    best_bin = explain("scorecard", models["scorecard"], d.X_train, d.categorical).contributions.min()
    rc_c = reason_codes(full[champion].contributions.iloc[both])
    rc_s = reason_codes(sc_contrib.iloc[both], reference=best_bin)

    def freq(codes: list[list[str]]) -> dict[str, float]:
        s = pd.Series([c[0] for c in codes if c]).value_counts(normalize=True).head(10)
        return {str(k): float(v) for k, v in s.items()}

    out["reason_codes"] = {
        "population": "test applicants rejected by both champion and scorecard at the valid-fixed cutoff",
        "n_rejected_champion": int(rej_c.sum()),
        "n_rejected_scorecard": int(rej_s.sum()),
        "n_rejected_both": len(both),
        "rejection_overlap_jaccard": float(len(both) / max(1, int((rej_c | rej_s).sum()))),
        "agreement": reason_agreement(rc_c, rc_s),
        "share_with_3_codes": {
            champion: float(np.mean([len(c) == 3 for c in rc_c])),
            "scorecard": float(np.mean([len(c) == 3 for c in rc_s])),
        },
        "top1_reason_frequency": {champion: freq(rc_c), "scorecard": freq(rc_s)},
    }
    (EXP / "explain.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"champion_codes": rc_c, "scorecard_codes": rc_s}, index=both).to_parquet(
        ART / "reason_codes_rejected_both.parquet"
    )
    print(
        json.dumps(
            {k: out[k] for k in ("champion", "importance_rank_corr_vs_champion", "reason_codes")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
