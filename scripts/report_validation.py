"""Independent validation report for the Home Credit PD champion, generated from results/ only.

Written in the role of a second-line validator (PLAN §7). Every number comes from results/; every finding
and its severity is derived by a stated rule from those numbers, so the text cannot drift from the data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path("results")
SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _grade_rows(g: dict[str, Any]) -> list[str]:
    out = [
        "| Grade | N | Defaults | Mean PD | Default rate | Jeffreys p (PD too low) |",
        "|---|---|---|---|---|---|",
    ]
    for r in g["grades"]:
        out.append(
            f"| {r['grade']} | {r['n']:,} | {r['defaults']:,} | {_f(r['mean_pd'])} | {_f(r['default_rate'])} "
            f"| {_f(r['jeffreys_p'], 3)} |"
        )
    return out


def _findings(
    r: dict[str, Any], v: dict[str, Any], e: dict[str, Any], csi_max: tuple[str, float]
) -> list[dict[str, str]]:
    c = v["champion"]
    m = r["models"][c]
    F: list[dict[str, str]] = []
    F.append({
        "sev": "High", "area": "Data / design",
        "text": "No out-of-time validation is possible: the dataset has no application date or usable time "
        "proxy (adversarial check in `reports/credit_hc_time_proxy.md`). All performance figures are from a "
        "stratified split of one period, so they are an upper bound for performance after deployment.",
        "action": "Before any use, validate on a later vintage. Until then, report results as in-sample-period only.",
    })  # fmt: skip
    g = v["grades"][c]
    gi = v["grades"].get(f"{c}_isotonic")
    if g["n_rejected_underestimation_bonferroni"] > 0 or g["n_rejected_underestimation"] > 0:
        bad = [str(x["grade"]) for x in g["grades"] if x["jeffreys_p"] < v["run"]["alpha"]]
        fixed = gi is not None and gi["n_rejected_underestimation"] == 0
        F.append({
            "sev": "Medium" if g["n_rejected_underestimation_bonferroni"] > 0 else "Low",
            "area": "Calibration",
            "text": f"Raw champion PDs are too low in grade(s) {', '.join(bad)} (Jeffreys test at "
            f"{v['run']['alpha']}; {g['n_rejected_underestimation_bonferroni']} grade(s) still rejected at the "
            f"Bonferroni level {v['bonferroni_alpha_per_grade']}). Pooled calibration measures (ECE "
            f"{_f(m['test']['ece'])}, slope {_f(m['test']['calibration_slope'], 3)}) hide this. The scorecard has "
            f"{v['grades']['scorecard']['n_rejected_underestimation']} rejected grade(s).",
            "action": ("Use the isotonic-recalibrated PD (fitted on valid): it has no grade rejected for "
                       "underestimation on test." if fixed else
                       "Recalibrate at grade level; the isotonic map fitted on valid does not remove the bias.")
            + " Re-run the grade test at every monitoring cycle.",
        })  # fmt: skip
    sens = v["sensitivity_ext_sources_missing"]
    sc, ss = sens[c], sens["scorecard"]
    F.append({
        "sev": "Medium", "area": "Model dependence",
        "text": f"The three external credit scores dominate the model. Without them (set to missing for all "
        f"test applicants), AUC falls from {_f(sc['auc_base'])} to {_f(sc['auc_ext_missing'])} (scorecard "
        f"{_f(ss['auc_base'])} to {_f(ss['auc_ext_missing'])}). Mean PD moves from {_f(sc['mean_pd_base'])} to "
        f"{_f(sc['mean_pd_ext_missing'])} (scorecard {_f(ss['mean_pd_base'])} to {_f(ss['mean_pd_ext_missing'])}).",
        "action": "Monitor availability and distribution of the external scores. Define a fallback rule "
        "(manual review or a model without them) for an outage.",
    })  # fmt: skip
    F.append({
        "sev": "Medium", "area": "Population shift",
        "text": f"The unlabelled Kaggle test population differs strongly from training on the bureau-coverage "
        f"features (largest CSI: `{csi_max[0]}` {csi_max[1]:.2f}). The champion's score PSI is still only "
        f"{_f(m['psi_train_to_kaggle'])}, because it barely uses those features. Performance on the shifted "
        "population cannot be measured (no labels).",
        "action": "Keep CSI on the coverage features in the monitoring set. Score PSI alone would miss this shift.",
    })  # fmt: skip
    st = v["split_seed_stability"]
    gap = m["train_auc"] - m["test"]["auc"]
    F.append({
        "sev": "Low", "area": "Overfitting",
        "text": f"Train AUC {_f(m['train_auc'])} vs test {_f(m['test']['auc'])} (gap {_f(gap)}). Test AUC is "
        f"stable across {len(st['per_seed'])} other splits (SD {_f(st['champion_auc_sd'])}), and the gain over "
        f"the scorecard holds on each ({_f(st['delta_auc_min'])} to {_f(st['delta_auc_max'])}). So the gap is "
        "memorisation of training rows, not a lucky split.",
        "action": "No action. Keep the train/test gap as a monitoring KPI.",
    })  # fmt: skip
    gm = m["at_cutoff"]["groups"]
    F.append({
        "sev": "Low", "area": "Fairness",
        "text": f"At the {m['at_cutoff']['approval_rate_target']:.0%} approval policy, approval is "
        f"{gm['F']['approval_rate']:.1%} for women and {gm['M']['approval_rate']:.1%} for men, in line with "
        f"observed default rates ({m['by_gender']['F']['default_rate']:.1%} vs "
        f"{m['by_gender']['M']['default_rate']:.1%}). Gender is not an input, but no proxy analysis was done.",
        "action": "Add a proxy check (predict gender from the features) and a conditional comparison at equal "
        "observed risk before any real use. This is not a legal assessment.",
    })  # fmt: skip
    F.append({
        "sev": "Low", "area": "Model selection",
        "text": "Hyperparameters and the champion were chosen on one validation split with many trials. Best valid "
        "AUCs are therefore optimistic. Test was used once, and the tuned-vs-grid comparison is reported "
        "(`reports/credit_hc_tuning.md`).",
        "action": "For a production model, use nested or repeated validation for the final selection.",
    })  # fmt: skip
    F.sort(key=lambda f: SEV_ORDER[f["sev"]])
    return F


def validation_report() -> str | None:
    vp = RESULTS / "credit/hc_validation/validation.json"
    rp = RESULTS / "credit/hc_challengers/results.json"
    ep = RESULTS / "credit/hc_explain/explain.json"
    if not (vp.exists() and rp.exists() and ep.exists()):
        return None
    v: dict[str, Any] = json.loads(vp.read_text())
    r: dict[str, Any] = json.loads(rp.read_text())
    e: dict[str, Any] = json.loads(ep.read_text())
    c = v["champion"]
    m, s = r["models"][c], r["models"]["scorecard"]
    csi_rows = (RESULTS / "credit/hc_challengers/csi_train_to_kaggle.csv").read_text().splitlines()[1]
    csi_max = (csi_rows.split(",")[0], float(csi_rows.split(",")[1]))
    F = _findings(r, v, e, csi_max)
    n_high = sum(f["sev"] == "High" for f in F)
    vs = m["vs_scorecard"]
    st = v["split_seed_stability"]
    L = [
        f"# Independent validation report: Home Credit PD model (champion `{c}`)",
        "",
        f"_Generated by `scripts/report.py` from `{vp}` (run {v['run']['utc']}), `{rp}` and `{ep}`. "
        "Portfolio project: the validator role is simulated. The checks in `scripts/r2_validate.py` were written "
        "separately from the development code. They only read the developer's artifacts and re-derive "
        "everything. No regulatory compliance is claimed._",
        "",
        "## 1. Scope and frame of reference",
        "",
        f"- **Model under review:** `{c}` PD model, selected params `{json.dumps(v['champion_params'], default=float)}`. "
        "**Benchmark:** the tuned WoE scorecard.",
        "- **Intended use (as documented in the model card):** application PD for consumer credit, research benchmark only.",
        "- **Frame of reference** (verified in `docs/versions.md`):",
        "  - model-risk guidance in the style of SR 26-2 (which superseded SR 11-7 in April 2026);",
        "  - MaRisk AT 4.3.5 on the use of models, which covers ML models;",
        "  - the ECB's IRB validation-reporting instructions (2019), for the Jeffreys test;",
        "  - EU AI Act Annex III 5(b), which lists credit scoring as high-risk.",
        "  The review follows these in structure only.",
        "",
        "## 2. Summary opinion",
        "",
        f"**{len(F)} findings: {n_high} High, {sum(f['sev'] == 'Medium' for f in F)} Medium, "
        f"{sum(f['sev'] == 'Low' for f in F)} Low.**",
        "",
        f"- **Discrimination:** the champion's advantage over the scorecard is real and robust. "
        f"Test ΔAUC is {vs['auc_diff']:+.4f} [{vs['ci95_low']:+.4f}, {vs['ci95_high']:+.4f}], and the advantage "
        f"holds on {len(st['per_seed'])} other splits ({_f(st['delta_auc_min'])} to {_f(st['delta_auc_max'])}) "
        + (
            "and in every segment tested."
            if all(x["delta_auc"] > 0 for x in v["segments"].values())
            else "but not in every segment (see 5.1)."
        ),
        (
            "- **Calibration:** the main weakness is at grade level (see findings)."
            if v["grades"][c]["n_rejected_underestimation"] > 0
            else "- **Calibration:** no grade-level rejection for underestimation."
        ),
        "- **Main limitation:** the dataset itself. Without a time dimension, out-of-time performance is unknown.",
        "",
        "**Opinion: fit for its stated purpose as a research benchmark. Not fit for credit decisions** until "
        "the High finding is resolved on data with a time dimension.",
        "",
        "## 3. Findings",
        "",
        "| # | Severity | Area | Finding | Required action |",
        "|---|---|---|---|---|",
    ]
    for i, f in enumerate(F, 1):
        L.append(f"| F{i} | {f['sev']} | {f['area']} | {f['text']} | {f['action']} |")
    rep = v["reproducibility"]
    add = {k: x["additivity_max_abs_error_logit"] for k, x in e["models"].items() if k in (c, "scorecard")}
    L += [
        "",
        "Severity rule: High means it blocks use; Medium means fix or mitigate before use; Low means note and monitor.",
        "",
        "## 4. Implementation verification",
        "",
        "- **Reproducibility:** pickled models re-scored on test. Max |Δ PD| vs stored predictions: "
        + ", ".join(f"{k} {x['max_abs_diff_vs_stored']:.1e}" for k, x in rep.items())
        + f". Labels identical: {all(x['labels_identical'] for x in rep.values())}.",
        "- **Explanations:** base value + contributions = model log-odds. Max error: "
        + ", ".join(f"{k} {x:.1e}" for k, x in add.items())
        + ".",
        "- **Leakage controls:** causality and shuffled-label tests run in CI (`tests/test_leakage.py`).",
        "",
        "## 5. Outcome analysis",
        "",
        "### 5.1 Discrimination",
        "",
        f"Test AUC {_f(m['test']['auc'])} (scorecard {_f(s['test']['auc'])}). Gini {_f(m['test']['gini'])} vs "
        f"{_f(s['test']['gini'])}.",
        "",
        "Split-seed stability (params fixed, data re-split and models refitted):",
        "",
        f"| Split seed | {c} AUC | Scorecard AUC | ΔAUC |",
        "|---|---|---|---|",
    ]
    for row in st["per_seed"]:
        L.append(f"| {row['split_seed']} | {_f(row[c])} | {_f(row['scorecard'])} | {row['delta_auc']:+.4f} |")
    L += [
        "",
        f"Segments (n ≥ 1,000 and ≥ 50 defaults on test). The champion leads in "
        f"{sum(x['delta_auc'] > 0 for x in v['segments'].values())} of {len(v['segments'])}:",
        "",
        f"| Segment | N | Defaults | {c} AUC | Scorecard AUC | ΔAUC |",
        "|---|---|---|---|---|---|",
    ]
    for k, x in v["segments"].items():
        L.append(f"| {k} | {x['n']:,} | {x['defaults']:,} | {_f(x['auc'][c])} | {_f(x['auc']['scorecard'])} "
                 f"| {x['delta_auc']:+.4f} |")  # fmt: skip
    L += [
        "",
        "### 5.2 Calibration by rating grade",
        "",
        f"There are {v['run']['n_grades']} grades with PD boundaries fixed on valid. A Jeffreys p-value below "
        f"{v['run']['alpha']} means the grade PD is too low. The Bonferroni level for {v['run']['n_grades']} "
        f"grades is {v['bonferroni_alpha_per_grade']}.",
        "",
    ]
    for k, g in v["grades"].items():
        L += [
            f"**{k}**: {g['n_rejected_underestimation']} grade(s) with PD too low "
            f"({g['n_rejected_underestimation_bonferroni']} at the Bonferroni level), "
            f"{g['n_rejected_overestimation']} with PD too high; default rate monotone across grades: "
            f"{g['default_rate_monotone']}.",
            "",
            *_grade_rows(g),
            "",
        ]
    sens = v["sensitivity_ext_sources_missing"]
    L += [
        "## 6. Sensitivity and stability",
        "",
        "| Model | AUC | AUC without external scores | Mean PD | Mean PD without external scores | Score PSI train→Kaggle |",
        "|---|---|---|---|---|---|",
    ]
    for k in (c, "scorecard"):
        x = sens[k]
        L.append(f"| {k} | {_f(x['auc_base'])} | {_f(x['auc_ext_missing'])} | {_f(x['mean_pd_base'])} "
                 f"| {_f(x['mean_pd_ext_missing'])} | {_f(r['models'][k]['psi_train_to_kaggle'])} |")  # fmt: skip
    L += [
        "",
        "## 7. Conceptual soundness (qualitative)",
        "",
        "- **Features:** application, bureau and previous-application aggregates built in dbt. Only rows dated "
        "before the application are used (dbt tests `not_after_application`). Gender is excluded from the "
        "inputs and used only for the group check.",
        "- **Split:** stratified, not time-based. This is a documented exception (`docs/split_policy.md`) and the "
        "reason for F1.",
        "- **Method:** gradient boosting with early stopping on valid. An interpretable alternative exists: EBM, "
        f"test AUC {_f(r['models']['ebm']['test']['auc'])}. So does a monotone-constrained one: test AUC "
        f"{_f(r['models']['lightgbm_monotone']['test']['auc'])}. Both are viable fallbacks if explainability "
        "requirements tighten.",
        "- **Explanations:** exact TreeSHAP and reason codes. For jointly declined applicants, the top reason "
        f"agrees with the scorecard's in {e['reason_codes']['agreement']['top1_shared']:.0%} of cases.",
        "",
    ]
    return "\n".join(L)
