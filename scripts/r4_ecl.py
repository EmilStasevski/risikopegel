"""R4: macro-conditional PD (satellite model) and a simplified IFRS 9-style ECL under scenarios.

1. Satellite: 12-month log change of German bankruptcy declarations on 12-month changes of unemployment
   and log industrial production (+ COVInsAG dummy), OLS with HAC standard errors.
2. Macro scenarios (3 years): baseline = ETS forecast (Module C model class) from the last observation;
   adverse = baseline + largest historical 12-month deterioration of each driver.
3. PD path: PD_k = PD_12m * exp(b . (x_k - x_0)) per scenario year k (log-elasticity carried over from
   business bankruptcies to the retail book: an assumption, stated in the report).
4. ECL: stage 1 = 12-month ECL, stage 2 = lifetime ECL over the remaining term (config/ecl.toml),
   scenario-weighted.

Writes results/credit/ecl/ecl.json and results/credit/ecl/pd_paths.csv (used by Module E).
"""

from __future__ import annotations

import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import ART, OUT, SEED

from risikopegel.credit.hc import load_mart, prepare
from risikopegel.forecast import models as fm_cls
from risikopegel.forecast.data import load_panel

ECL = Path("results/credit/ecl")
CFG = Path("config/ecl.toml")


def satellite(panel: dict[str, pd.Series], cfg: dict[str, Any]) -> dict[str, Any]:
    t, (u, ip) = cfg["target"], cfg["drivers"]
    df = pd.DataFrame({"b": np.log(panel[t]), "u": panel[u], "ip": np.log(panel[ip])}).dropna()
    d = df.diff(12).dropna()
    covid = pd.period_range(cfg["covid_start"], cfg["covid_end"], freq="M")
    X = pd.DataFrame(
        {"d12_unemp_pp": d["u"], "d12_log_ip": d["ip"], "covinsag": d.index.isin(covid).astype(float)}
    )
    X = sm.add_constant(X)
    res = sm.OLS(d["b"], X).fit(cov_type="HAC", cov_kwds={"maxlags": cfg["hac_lags"]})
    ci = res.conf_int(0.05)
    return {
        "n": int(res.nobs),
        "period": f"{d.index[0]}..{d.index[-1]}",
        "r2": float(res.rsquared),
        "coef": {k: float(v) for k, v in res.params.items()},
        "se_hac": {k: float(v) for k, v in res.bse.items()},
        "p": {k: float(v) for k, v in res.pvalues.items()},
        "ci95": {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in res.params.index},
    }


def scenarios(panel: dict[str, pd.Series], cfg: dict[str, Any], years: int) -> dict[str, Any]:
    u, ip = cfg["drivers"]
    out: dict[str, Any] = {}
    paths = {}
    for k, transform in ((u, lambda x: x), (ip, np.log)):
        s = transform(panel[k])
        y = s.to_numpy()
        q = fm_cls.ets(y, fm_cls.SpecCache(), k, reselect=True, h=12 * years)
        base = q[[12 * (i + 1) - 1 for i in range(years)], 1]
        d12 = s.diff(12).dropna()
        worst = float(d12.max()) if k == u else float(d12.min())  # unemployment up, production down
        when = str(d12.idxmax() if k == u else d12.idxmin())
        paths[k] = {
            "last": float(y[-1]),
            "last_period": str(s.index[-1]),
            "baseline": base.tolist(),
            "adverse": (base + worst).tolist(),
            "worst_12m_change": worst,
            "worst_12m_month": when,
        }
    out["drivers"] = paths
    return out


def pd_multipliers(
    sat: dict[str, Any], sc: dict[str, Any], cfg: dict[str, Any], years: int
) -> dict[str, list[float]]:
    u, ip = cfg["drivers"]
    b_u, b_ip = sat["coef"]["d12_unemp_pp"], sat["coef"]["d12_log_ip"]
    out = {}
    for name in ("baseline", "adverse"):
        du = np.array(sc["drivers"][u][name]) - sc["drivers"][u]["last"]
        dip = np.array(sc["drivers"][ip][name]) - sc["drivers"][ip]["last"]
        out[name] = np.exp(b_u * du + b_ip * dip)[:years].tolist()
    return out


def ecl_table(
    pd12: np.ndarray, ead: np.ndarray, term_m: np.ndarray, mult: list[float], cfg: dict[str, Any]
) -> dict[str, Any]:
    lgd = cfg["portfolio"]["lgd"]
    years = len(mult)
    pdk = np.clip(pd12[:, None] * np.array(mult)[None, :], 0, 0.999)  # (n, years)
    # lifetime PD over the remaining term: full years, then a pro-rata share of the last (partial) year
    surv = np.ones_like(pd12)
    life = np.zeros_like(pd12)
    remaining = term_m / 12.0
    for k in range(int(np.ceil(remaining.max()))):
        frac = np.clip(remaining - k, 0, 1)
        p = pdk[:, min(k, years - 1)] * frac
        life += surv * p
        surv *= 1 - p
    stage2 = pd12 >= cfg["staging"]["stage2_pd_threshold"]
    ecl12 = pdk[:, 0] * lgd * ead
    ecl_life = life * lgd * ead
    ecl = np.where(stage2, ecl_life, ecl12)
    return {
        "ecl_total": float(ecl.sum()),
        "ead_total": float(ead.sum()),
        "coverage_pct": float(100 * ecl.sum() / ead.sum()),
        "stage1": {
            "n": int((~stage2).sum()),
            "ead": float(ead[~stage2].sum()),
            "ecl": float(ecl[~stage2].sum()),
        },
        "stage2": {"n": int(stage2.sum()), "ead": float(ead[stage2].sum()), "ecl": float(ecl[stage2].sum())},
        "mean_pd_by_year": pdk.mean(axis=0).tolist(),
        "ead_weighted_pd_by_year": (pdk * ead[:, None]).sum(axis=0).__truediv__(ead.sum()).tolist(),
        "mean_lifetime_pd": float(life.mean()),
    }


def main() -> None:
    ECL.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.loads(CFG.read_text())
    years = cfg["scenarios"]["years"]
    panel = load_panel()
    sat = satellite(panel, cfg["satellite"])
    sc = scenarios(panel, cfg["satellite"], years)
    mult = pd_multipliers(sat, sc, cfg["satellite"], years)

    res = json.loads((OUT / "results.json").read_text())
    champ = res["champion"]["name"]
    d = prepare(load_mart(), seed=SEED)
    pred = pd.read_parquet(ART / "test_predictions.parquet")
    pd12 = pred[champ].to_numpy(dtype=float)
    ead = d.X_test[cfg["portfolio"]["ead"]].to_numpy(dtype=float)
    term = d.X_test[cfg["portfolio"]["term_from"]].fillna(d.X_test[cfg["portfolio"]["term_from"]].median())
    term_m = np.clip(
        term.to_numpy(dtype=float), cfg["portfolio"]["term_min_months"], cfg["portfolio"]["term_max_months"]
    )

    tables = {"current_macro": ecl_table(pd12, ead, term_m, [1.0] * years, cfg)}
    for name in ("baseline", "adverse"):
        tables[name] = ecl_table(pd12, ead, term_m, mult[name], cfg)
    w = cfg["scenarios"]["weights"]
    weighted = sum(w[k] * tables[k]["ecl_total"] for k in w)
    out = {
        "run": {"script": "scripts/r4_ecl.py", "utc": datetime.now(UTC).isoformat(timespec="seconds")},
        "config": cfg,
        "champion": champ,
        "portfolio": {
            "n": len(pd12),
            "observed_default_rate": float(pred["y"].mean()),
            "mean_pd": float(pd12.mean()),
            "ead_total": float(ead.sum()),
            "term_months_median": float(np.median(term_m)),
        },
        "satellite": sat,
        "scenarios": sc,
        "pd_multiplier_by_year": mult,
        "ecl": tables,
        "ecl_weighted": {
            "weights": w,
            "ecl_total": weighted,
            "coverage_pct": 100 * weighted / float(ead.sum()),
        },
    }
    (ECL / "ecl.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(
        [
            {"scenario": k, "year": i + 1, "pd_multiplier": v}
            for k, vs in mult.items()
            for i, v in enumerate(vs)
        ]
    ).to_csv(ECL / "pd_paths.csv", index=False)
    print(json.dumps({k: out[k] for k in ("satellite", "pd_multiplier_by_year", "ecl_weighted")}, indent=2))
    for k, t in tables.items():
        print(k, round(t["coverage_pct"], 3), t["stage2"]["n"], [round(x, 4) for x in t["mean_pd_by_year"]])


if __name__ == "__main__":
    main()
