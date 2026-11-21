"""Module E: simplified Risikotragfähigkeit for a stylised bank (config/rtf.toml), tied to Modules A-C.

Economic perspective (1 year, 99.9%): credit (Monte Carlo on Module A PDs, IRB benchmark), market
(Module B FHS VaR, scaled), IRRBB (historical simulation of 1-year ECB AAA curve changes; EU supervisory
shocks as the regulatory view), operational risk (CRR BIC as proxy); aggregation and limit use.
Normative perspective (3 years): CET1 ratio path under the R4 baseline/adverse PD paths.
Reverse stress tests: the PD multiplier that breaches the CET1 requirement; the parallel rate shock that
uses up the risk cover.

Writes results/rtf/rtf.json and results/rtf/pd_buckets.csv (aggregate PD buckets, used by the API).
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import ART, OUT, SEED

from risikopegel.credit.hc import load_mart, prepare
from risikopegel.market.data import load_portfolio
from risikopegel.rtf import credit, irrbb
from risikopegel.sources import load_catalog, load_raw

RES = Path("results/rtf")
CFG = Path("config/rtf.toml")
TENORS = {
    "yc_aaa_3m": 0.25,
    "yc_aaa_1y": 1.0,
    "yc_aaa_2y": 2.0,
    "yc_aaa_5y": 5.0,
    "yc_aaa_10y": 10.0,
    "yc_aaa_20y": 20.0,
    "yc_aaa_30y": 30.0,
}


def load_book(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    champ = json.loads((OUT / "results.json").read_text())["champion"]["name"]
    pd_ = pd.read_parquet(ART / "test_predictions.parquet")[champ].to_numpy(dtype=float)
    amt = prepare(load_mart(), seed=SEED).X_test["credit_amount"].to_numpy(dtype=float)
    ead = amt / amt.sum() * cfg["credit_book"]["ead_total"]
    return np.maximum(pd_, cfg["credit_book"]["pd_floor"]), ead


def curves() -> pd.DataFrame:
    cat = {s.id: s for s in load_catalog()}
    cols = {}
    for sid, tenor in TENORS.items():
        df = load_raw(cat[sid])
        cols[tenor] = pd.Series(df["value"].to_numpy() / 100, index=pd.to_datetime(df["period"]))
    return pd.DataFrame(cols).dropna()


def eve(cfg: dict[str, Any], t: np.ndarray, r: np.ndarray) -> float:
    c = cfg["irrbb"]
    a = irrbb.cash_flows(
        cfg["other_assets"]["fixed_rate_assets"], c["asset_tenors"], c["asset_shares"], c["asset_coupon"]
    )
    d = irrbb.cash_flows(c["deposits"], c["deposit_tenors"], c["deposit_shares"], c["deposit_rate"])
    w = irrbb.cash_flows(c["wholesale"], c["wholesale_tenors"], c["wholesale_shares"], c["wholesale_rate"])
    return irrbb.pv(*a, t, r) - irrbb.pv(*d, t, r) - irrbb.pv(*w, t, r)


def delta_nii(cfg: dict[str, Any], dr: float) -> float:
    """1-year NII change for a parallel shift, constant balance sheet: items repricing within the year earn
    or pay the shift for the rest of the year (simple repricing gap)."""
    c = cfg["irrbb"]
    out = 0.0
    for notional, tenors, shares, sign in (
        (cfg["other_assets"]["fixed_rate_assets"], c["asset_tenors"], c["asset_shares"], 1),
        (c["deposits"], c["deposit_tenors"], c["deposit_shares"], -1),
        (c["wholesale"], c["wholesale_tenors"], c["wholesale_shares"], -1),
    ):
        for T, s in zip(tenors, shares, strict=True):
            if T <= 1:
                out += sign * notional * s * dr * (1 - T)
    # the consumer credit book is assumed fixed-rate (it reprices only through loan turnover)
    return out


def irrbb_block(cfg: dict[str, Any], cv: pd.DataFrame) -> dict[str, Any]:
    t = np.array(cv.columns, dtype=float)
    base = cv.iloc[-1].to_numpy()
    eve0 = eve(cfg, t, base)
    sup = {s: eve(cfg, t, irrbb.apply_shock(t, base, s)) - eve0 for s in irrbb.SCENARIOS}
    worst = min(sup, key=lambda k: sup[k])
    tier1 = cfg["bank"]["cet1"]
    nii = {"parallel_up": delta_nii(cfg, 0.02), "parallel_down": delta_nii(cfg, -0.02)}
    # historical simulation: every 1-year change of the curve since 2004 applied to today's curve
    hz = cfg["economic"]["irrbb_hist_horizon_days"]
    dcv = (cv.shift(-hz) - cv).dropna().to_numpy()
    hist = np.array([eve(cfg, t, base + dd) - eve0 for dd in dcv[::5]])  # weekly-spaced windows
    conf = cfg["economic"]["confidence"]
    return {
        "curve_date": str(cv.index[-1].date()),
        "curve": dict(zip(map(str, t), base.tolist(), strict=True)),
        "eve_base": eve0,
        "delta_eve_supervisory": sup,
        "worst_scenario": worst,
        "worst_delta_eve": sup[worst],
        "sot_eve_ratio_tier1": -sup[worst] / tier1,
        "sot_eve_outlier": bool(-sup[worst] > 0.15 * tier1),
        "delta_nii_parallel": nii,
        "sot_nii_ratio_tier1": -min(nii.values()) / tier1,
        "sot_nii_outlier": bool(-min(nii.values()) > 0.05 * tier1),
        "hist_n_windows": int(hist.size),
        "hist_first_window": str(cv.index[0].date()),
        "hist_loss_quantile": float(-np.quantile(hist, 1 - conf)),
        "hist_worst_loss": float(-hist.min()),
    }


def market_block(cfg: dict[str, Any]) -> dict[str, Any]:
    f = pd.read_parquet("data/processed/market/forecasts.parquet")
    f = f[
        (f["portfolio"] == "main")
        & (f["model"] == "fhs_garch_t")
        & (f["window"] == 1000)
        & (f["level"] == 0.99)
    ]
    last = f.iloc[-1]
    tb = cfg["other_assets"]["trading_book"]
    conf = cfg["economic"]["confidence"]
    k = stats.norm.ppf(conf) / stats.norm.ppf(0.99)
    var1d = float(last["var"]) / 100 * tb
    p = load_portfolio()["portfolio"]
    worst_1y = float(np.expm1(np.log1p(p).rolling(250).sum()).min())
    return {
        "var99_1d_pct": float(last["var"]),
        "var_date": str(last["date"].date()),
        "ec": var1d * np.sqrt(cfg["economic"]["market_holding_days"]) * k,
        "ec_60d": var1d * np.sqrt(60) * k,
        "own_funds_simplified": 3 * var1d * np.sqrt(10),  # 3 x 10-day VaR99 (simplified, assumption)
        "worst_1y_return_pct": 100 * worst_1y,
    }


def aggregate(ec: dict[str, float], cfg: dict[str, Any]) -> dict[str, Any]:
    keys = ["credit", "market", "irrbb", "op"]
    c = cfg["economic"]["corr"]
    C = np.eye(4)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                C[i, j] = C[j, i] = c[f"{a}_{b}"]
    e = np.array([ec[k] for k in keys])
    cover = cfg["bank"]["cet1"] * cfg["economic"]["risk_cover_share_of_cet1"]
    total_sum, total_vc = float(e.sum()), float(np.sqrt(e @ C @ e))
    limits = {k: cfg["economic"]["limits"][k] * cover for k in keys}

    def light(u: float) -> str:
        return "green" if u < 0.8 else ("amber" if u <= 1.0 else "red")

    return {
        "ec": ec,
        "total_simple_sum": total_sum,
        "total_var_covar": total_vc,
        "diversification_pct": 100 * (1 - total_vc / total_sum),
        "risk_cover": cover,
        "utilisation_sum": total_sum / cover,
        "utilisation_var_covar": total_vc / cover,
        "traffic_light_total": light(total_sum / cover),
        "limits": {
            k: {"limit": limits[k], "use": ec[k] / limits[k], "light": light(ec[k] / limits[k])} for k in keys
        },
    }


def normative(  # noqa: PLR0917
    cfg: dict[str, Any],
    pd_: np.ndarray,
    ead: np.ndarray,
    mult: dict[str, list[float]],
    rwa_other: float,
    adverse_market_loss: float,
) -> dict[str, Any]:
    n = cfg["normative"]
    lgd = cfg["credit_book"]["lgd"]
    other = cfg["other_assets"]["fixed_rate_assets"] + cfg["other_assets"]["trading_book"]
    el_rate0 = float((pd_ * ead).sum() / ead.sum()) * lgd  # pricing is fixed at today's PDs
    ppp = (el_rate0 + n["credit_book_spread_over_el"]) * float(ead.sum()) + n[
        "other_assets_net_margin"
    ] * other
    req = sum(n["cet1_requirement"].values())
    out: dict[str, Any] = {
        "requirement": req,
        "target": req + n["management_buffer"],
        "paths": {},
        "pre_provision_profit": None,
    }
    for s, m in mult.items():
        cet1, path = cfg["bank"]["cet1"], []
        rwa0 = float((credit.irb_retail_rw(pd_, lgd) * ead).sum()) + rwa_other
        path.append({"year": 0, "cet1": cet1, "rwa": rwa0, "ratio": cet1 / rwa0})
        for k, mk in enumerate(m[: n["years"]], start=1):
            pk = np.clip(pd_ * mk, cfg["credit_book"]["pd_floor"], 0.999)
            loss = float((pk * ead).sum() * lgd)
            mkt = adverse_market_loss if (s == "adverse" and k == 1) else 0.0
            net = ppp - loss - mkt
            cet1 += net * (1 - n["payout_ratio"]) if net > 0 else net
            rwa = float((credit.irb_retail_rw(pk, lgd) * ead).sum()) + rwa_other
            path.append(
                {
                    "year": k,
                    "credit_loss": loss,
                    "market_loss": mkt,
                    "net_income": net,
                    "cet1": cet1,
                    "rwa": rwa,
                    "ratio": cet1 / rwa,
                }
            )
        out["paths"][s] = path
        out["pre_provision_profit"] = ppp
        out[f"min_ratio_{s}"] = min(p["ratio"] for p in path)
    return out


def reverse_pd(cfg: dict[str, Any], pd_: np.ndarray, ead: np.ndarray, rwa_other: float, req: float) -> float:
    """Constant PD multiplier (all 3 years) at which the minimum CET1 ratio equals the requirement."""

    def min_ratio(m: float) -> float:
        r = normative(cfg, pd_, ead, {"rev": [m] * cfg["normative"]["years"]}, rwa_other, 0.0)
        return float(r["min_ratio_rev"])

    lo, hi = 1.0, 50.0
    if min_ratio(hi) > req:
        return float("inf")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if min_ratio(mid) > req else (lo, mid)
    return (lo + hi) / 2


def reverse_rates(cfg: dict[str, Any], cv: pd.DataFrame, other_ec: float, cover: float) -> float | None:
    """Parallel upward shock (bp) at which IRRBB loss + other economic capital equals the risk cover."""
    t = np.array(cv.columns, dtype=float)
    base = cv.iloc[-1].to_numpy()
    e0 = eve(cfg, t, base)

    def loss(bp: float) -> float:
        return -(eve(cfg, t, base + bp / 1e4) - e0)

    if loss(2000) + other_ec < cover:
        return None
    lo, hi = 0.0, 2000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if loss(mid) + other_ec < cover else (lo, mid)
    return (lo + hi) / 2


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    cfg = tomllib.loads(CFG.read_text())
    eco = cfg["economic"]
    pd_, ead = load_book(cfg)
    lgd = cfg["credit_book"]["lgd"]

    # credit: IRB benchmark and Monte Carlo
    k_irb = credit.irb_retail_k(pd_, lgd)
    b = credit.make_buckets(pd_, ead, cfg["credit_book"]["n_buckets"])
    pd.DataFrame({"pd": b.pd, "n": b.n, "ead_share": b.ead / b.ead.sum()}).to_csv(
        RES / "pd_buckets.csv", index=False
    )
    losses = credit.simulate_losses(b, lgd, eco["n_sims"], eco["seed"])
    cr = credit.loss_stats(losses, eco["confidence"])
    sens = {}
    for f in (0.5, 1.5):
        rho = np.clip(credit.retail_other_correlation(b.pd) * f, 0, 0.99)
        sens[f"rho_x{f}"] = credit.loss_stats(
            credit.simulate_losses(b, lgd, eco["n_sims"], eco["seed"], rho), eco["confidence"]
        )
    sens["lgd_0.30"] = credit.loss_stats(
        credit.simulate_losses(b, 0.30, eco["n_sims"], eco["seed"]), eco["confidence"]
    )
    credit_block = {
        "n_loans": int(pd_.size),
        "ead": float(ead.sum()),
        "ead_weighted_pd": float((pd_ * ead).sum() / ead.sum()),
        "irb_k": float((k_irb * ead).sum()),
        "irb_rwa": float((12.5 * k_irb * ead).sum()),
        "irb_el": float((pd_ * lgd * ead).sum()),
        "mc": cr,
        "mc_sensitivity": sens,
        "mean_correlation": float(np.average(credit.retail_other_correlation(pd_), weights=ead)),
    }

    cv = curves()
    ir = irrbb_block(cfg, cv)
    mk = market_block(cfg)
    op = 0.12 * cfg["bank"]["business_indicator"] / 1000 * 1000  # BI <= 1 bn: BIC = 0.12 * BI (CRR Art. 313)
    ec = {"credit": cr["ul"], "market": mk["ec"], "irrbb": ir["hist_loss_quantile"], "op": op}
    agg = aggregate(ec, cfg)

    rwa_other = (
        cfg["other_assets"]["fixed_rate_assets"] * cfg["other_assets"]["fixed_rate_risk_weight"]
        + 12.5 * mk["own_funds_simplified"]
        + 12.5 * op
    )
    paths = pd.read_csv("results/credit/ecl/pd_paths.csv")
    mult: dict[str, list[float]] = {
        str(s): g.sort_values("year")["pd_multiplier"].tolist() for s, g in paths.groupby("scenario")
    }
    mult = {"current_macro": [1.0] * cfg["normative"]["years"], **mult}
    adverse_mkt = -mk["worst_1y_return_pct"] / 100 * cfg["other_assets"]["trading_book"]
    norm = normative(cfg, pd_, ead, mult, rwa_other, max(adverse_mkt, 0.0))
    rev_pd = reverse_pd(cfg, pd_, ead, rwa_other, norm["requirement"])
    rev_bp = reverse_rates(cfg, cv, ec["credit"] + ec["market"] + ec["op"], agg["risk_cover"])

    out = {
        "run": {"script": "scripts/r5_rtf.py", "utc": datetime.now(UTC).isoformat(timespec="seconds")},
        "config": cfg,
        "credit": credit_block,
        "market": mk,
        "irrbb": ir,
        "op": {"bic": op, "business_indicator": cfg["bank"]["business_indicator"]},
        "economic": agg,
        "normative": norm,
        "reverse_stress": {
            "pd_multiplier_breaching_cet1_requirement": rev_pd,
            "parallel_up_bp_using_up_risk_cover": rev_bp,
        },
    }
    (RES / "rtf.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("economic", "reverse_stress")}, indent=2, default=float))
    for s, p in norm["paths"].items():
        print(s, [round(x["ratio"] * 100, 2) for x in p])


if __name__ == "__main__":
    main()
