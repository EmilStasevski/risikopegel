"""Headline block for README.md (EN) and README.de.md (DE), filled from results/ only (PLAN §0 rule 1, §11).

report.py replaces the text between the GENERATED markers in both READMEs; everything else in them is prose
without numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

R = Path("results")
BEGIN, END = "<!-- BEGIN GENERATED: scripts/report_readme.py -->", "<!-- END GENERATED -->"


def _j(rel: str) -> dict[str, Any] | None:
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def facts() -> dict[str, Any] | None:
    ch, val = _j("credit/hc_challengers/results.json"), _j("credit/hc_validation/validation.json")
    bt, fc = _j("market/backtest/backtest.json"), _j("macro/forecast/forecast.json")
    ecl, rtf = _j("credit/ecl/ecl.json"), _j("rtf/rtf.json")
    ex, sig = _j("llm/explanations/summary.json"), _j("llm/text_signal/signal.json")
    if not (ch and val and bt and fc and ecl and rtf and ex and sig):
        return None
    champ = ch["champion"]["name"]
    c, s = ch["models"][champ], ch["models"]["scorecard"]
    py = pd.read_csv(R / "market/backtest/per_year.csv")
    m = py[(py["window"] == 1000) & (py["level"] == 0.99)]
    passed = m.assign(p=(m["p_uc"] > 0.05) & (m["p_cc"] > 0.05)).groupby("model")["p"].sum()
    summ = pd.read_csv(R / "market/backtest/summary.csv")
    by_h_df = pd.read_csv(R / "macro/forecast/by_h.csv")
    by_h = {
        (m_, h_): float(v) for m_, h_, v in zip(by_h_df["model"], by_h_df["h"], by_h_df["mase"], strict=True)
    }
    dm = pd.read_csv(R / "macro/forecast/dm.csv")
    tfm_vs_ets_h1 = dm[(dm["reference"] == "ets") & (dm["model"] == "timesfm25") & (dm["h"] == 1)].iloc[0]
    sig_best = min(
        (r["mse_ratio"], r["dm_p"], k, t) for k, rr in sig["results"].items() for t, r in rr.items()
    )
    return {
        "champ": champ,
        "gini_c": c["test"]["gini"],
        "gini_s": s["test"]["gini"],
        "dauc": c["vs_scorecard"]["auc_diff"],
        "dauc_lo": c["vs_scorecard"]["ci95_low"],
        "dauc_hi": c["vs_scorecard"]["ci95_high"],
        "slope_c": c["test"]["calibration_slope"],
        "slope_s": s["test"]["calibration_slope"],
        "ece_c": c["test"]["ece"],
        "ece_s": s["test"]["ece"],
        "psi_c": c["psi_train_to_kaggle"],
        "psi_s": s["psi_train_to_kaggle"],
        "n_models_var": int(summ["model"].nunique()),
        "first_fc": bt["data"]["first_forecast"],
        "last_ret": bt["data"]["last_return"],
        "years": int(m["year"].nunique()),
        "pass_fhs": int(passed["fhs_garch_t"]),
        "pass_hs": int(passed["historical"]),
        "pass_norm": int(passed["normal"]),
        "pass_ml": int(max(passed["lgbm_quantile"], passed["lgbm_vol_fhs"])),
        "n_series": len(fc["series"]),
        "n_models_fc": len(fc["compute"]),
        "mase_tfm_1": by_h[("timesfm25", 1)],
        "mase_ets_1": by_h[("ets", 1)],
        "dm_p_tfm": float(tfm_vs_ets_h1["pooled_p"]),
        "j_ets": fc["compute"]["ets"]["joules_above_idle_per_forecast"],
        "j_tfm": fc["compute"]["timesfm25"]["joules_above_idle_per_forecast"],
        "j_bolt": fc["compute"]["chronos_bolt_small"]["joules_above_idle_per_forecast"],
        "ecl_cov_base": ecl["ecl"]["baseline"]["coverage_pct"],
        "ecl_cov_adv": ecl["ecl"]["adverse"]["coverage_pct"],
        "rtf_use": rtf["economic"]["utilisation_sum"],
        "rtf_light": rtf["economic"]["traffic_light_total"],
        "rtf_min_adv": rtf["normative"]["min_ratio_adverse"],
        "rtf_req": rtf["normative"]["requirement"],
        "rev_pd": rtf["reverse_stress"]["pd_multiplier_breaching_cet1_requirement"],
        "irrbb_outlier": rtf["irrbb"]["sot_eve_outlier"],
        "faith_zero": ex["variants"]["zero_shot"]["pass_rate"],
        "faith_ver": ex["variants"]["verified_retry"]["pass_rate"],
        "ship_ver": ex["variants"]["verified_retry"]["shipped_faithful_rate"],
        "n_expl": ex["sample"]["n"],
        "sig_ratio": sig_best[0],
        "sig_p": sig_best[1],
        "sig_m": sig_best[2],
        "sig_t": sig_best[3],
    }


def block_en(f: dict[str, Any]) -> str:
    return "\n".join(
        [
            BEGIN,
            "",
            "| Module | Headline (all numbers generated from `results/`) |",
            "|---|---|",
            f"| A, credit PD (Home Credit) | {f['champ']} vs WoE scorecard: Gini {f['gini_c']:.3f} vs {f['gini_s']:.3f}, "
            f"ΔAUC {f['dauc']:+.3f} [{f['dauc_lo']:+.3f}, {f['dauc_hi']:+.3f}]; calibration slope {f['slope_c']:.2f} vs "
            f"{f['slope_s']:.2f}, ECE {f['ece_c']:.4f} vs {f['ece_s']:.4f}; score PSI {f['psi_c']:.4f} vs {f['psi_s']:.4f} |",
            f"| B, VaR/ES backtesting | {f['n_models_var']} models, ECB FX/rates portfolio, {f['first_fc']} to {f['last_ret']}: "
            f"99% VaR passes Kupiec + Christoffersen in {f['pass_fhs']}/{f['years']} years for FHS GARCH-t vs "
            f"{f['pass_hs']} (historical simulation), {f['pass_norm']} (normal), at most {f['pass_ml']} (ML) |",
            f"| C, macro forecasting | {f['n_series']} Eurostat series, {f['n_models_fc']} models incl. Chronos-2, Chronos-Bolt, "
            f"TimesFM 2.5: h=1 MASE TimesFM {f['mase_tfm_1']:.3f} vs ETS {f['mase_ets_1']:.3f} (DM p = {f['dm_p_tfm']:.2f}); "
            f"energy per forecast ETS {f['j_ets']:.2f} J, TimesFM {f['j_tfm']:.2f} J, Chronos-Bolt {f['j_bolt']:.2f} J |",
            f"| A+C, scenario ECL | macro-conditional PD: ECL coverage {f['ecl_cov_base']:.2f}% (baseline) vs "
            f"{f['ecl_cov_adv']:.2f}% (adverse) |",
            f"| E, Risikotragfähigkeit | stylised bank: risk-cover use {f['rtf_use']:.0%} ({f['rtf_light']}); adverse "
            f"minimum CET1 ratio {f['rtf_min_adv']:.1%} vs {f['rtf_req']:.2%} required; reverse stress: PD × {f['rev_pd']:.2f}; "
            f"IRRBB EVE outlier: {'yes' if f['irrbb_outlier'] else 'no'} |",
            f"| D, LLM explanations | {f['n_expl']} German explanations: {f['faith_zero']:.0%} faithful zero-shot, "
            f"{f['faith_ver']:.0%} with verifier + retry, {f['ship_ver']:.0%} of shipped texts verified |",
            f"| D, ECB tone signal | best out-of-sample MSE ratio {f['sig_ratio']:.3f} ({f['sig_m']}, {f['sig_t']}; DM p = "
            f"{f['sig_p']:.2f}) |",
            "",
            END,
        ]
    )


def block_de(f: dict[str, Any]) -> str:
    def pc(x: float, nd: int = 0) -> str:
        return f"{100 * x:.{nd}f} %".replace(".", ",")

    def nf(x: float, nd: int = 3) -> str:
        return f"{x:.{nd}f}".replace(".", ",")

    return "\n".join(
        [
            BEGIN,
            "",
            "| Modul | Kernergebnis (alle Zahlen aus `results/` erzeugt) |",
            "|---|---|",
            f"| A, Kredit-PD (Home Credit) | {f['champ']} vs. WoE-Scorecard: Gini {nf(f['gini_c'])} vs. {nf(f['gini_s'])}, "
            f"ΔAUC {nf(f['dauc'])} [{nf(f['dauc_lo'])}; {nf(f['dauc_hi'])}]; Kalibrierungssteigung {nf(f['slope_c'], 2)} vs. "
            f"{nf(f['slope_s'], 2)}, ECE {nf(f['ece_c'], 4)} vs. {nf(f['ece_s'], 4)}; Score-PSI {nf(f['psi_c'], 4)} vs. {nf(f['psi_s'], 4)} |",
            f"| B, VaR/ES-Backtesting | {f['n_models_var']} Modelle, EZB-FX/Zins-Portfolio, {f['first_fc']} bis {f['last_ret']}: "
            f"99-%-VaR besteht Kupiec + Christoffersen in {f['pass_fhs']}/{f['years']} Jahren mit FHS GARCH-t vs. "
            f"{f['pass_hs']} (historische Simulation), {f['pass_norm']} (Normalverteilung), höchstens {f['pass_ml']} (ML) |",
            f"| C, Makroprognosen | {f['n_series']} Eurostat-Reihen, {f['n_models_fc']} Modelle inkl. Chronos-2, Chronos-Bolt, "
            f"TimesFM 2.5: MASE h=1 TimesFM {nf(f['mase_tfm_1'])} vs. ETS {nf(f['mase_ets_1'])} (DM p = {nf(f['dm_p_tfm'], 2)}); "
            f"Energie je Prognose ETS {nf(f['j_ets'], 2)} J, TimesFM {nf(f['j_tfm'], 2)} J, Chronos-Bolt {nf(f['j_bolt'], 2)} J |",
            f"| A+C, Szenario-ECL | makrobedingte PD: ECL-Deckung {nf(f['ecl_cov_base'], 2)} % (Basis) vs. "
            f"{nf(f['ecl_cov_adv'], 2)} % (adverses Szenario) |",
            f"| E, Risikotragfähigkeit | Musterbank: Auslastung des Risikodeckungspotenzials {pc(f['rtf_use'])} ({f['rtf_light']}); "
            f"minimale harte Kernkapitalquote adverses Szenario {pc(f['rtf_min_adv'], 1)} vs. {pc(f['rtf_req'], 2)} Anforderung; "
            f"inverser Stresstest: PD × {nf(f['rev_pd'], 2)}; IRRBB-Ausreißer (EVE): {'ja' if f['irrbb_outlier'] else 'nein'} |",
            f"| D, LLM-Erklärungen | {f['n_expl']} deutsche Erklärungen: {pc(f['faith_zero'])} faithful (zero-shot), "
            f"{pc(f['faith_ver'])} mit Verifier + Retry, {pc(f['ship_ver'])} der ausgelieferten Texte verifiziert |",
            f"| D, EZB-Tonalität | bestes Out-of-Sample-MSE-Verhältnis {nf(f['sig_ratio'])} ({f['sig_m']}, {f['sig_t']}; "
            f"DM p = {nf(f['sig_p'], 2)}) |",
            "",
            END,
        ]
    )


def update_readmes() -> list[str]:
    f = facts()
    if f is None:
        return []
    done = []
    for path, block in ((Path("README.md"), block_en(f)), (Path("README.de.md"), block_de(f))):
        if not path.exists():
            continue
        s = path.read_text()
        if BEGIN in s and END in s:
            s = s[: s.index(BEGIN)] + block + s[s.index(END) + len(END) :]
            path.write_text(s)
            done.append(str(path))
    return done
