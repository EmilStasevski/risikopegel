"""R3: rolling one-day VaR/ES forecasts for 9 models x 3 windows, full backtest suite.

Protocol (fixed before the run):
- portfolio and regimes: config/market.toml; returns in percent, loss = -return
- every forecast for day t uses r[t-window:t] only; all windows start forecasting on the same day
  (index = largest window), so every model/window is scored on the same days
- main window 1000 days, sensitivity 250/500; ML challenger refitted every 20 days
- tests: Kupiec, Christoffersen (ind., cond. coverage), Basel traffic light per calendar year and on
  rolling 250-day windows, Acerbi-Szekely Z2 and an exceedance-residual test for ES, pinball and FZ0 loss,
  Diebold-Mariano vs FHS, Model Confidence Set (arch)

Writes results/market/backtest/*; forecasts go to data/processed/market/forecasts.parquet.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

import arch
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from risikopegel.market import backtest as bt
from risikopegel.market.data import load_config, load_portfolio
from risikopegel.market.ml import lgbm_quantile_forecasts, lgbm_vol_forecasts
from risikopegel.market.var_models import MODELS, rolling_forecasts

OUT = Path("results/market/backtest")
ART = Path("data/processed/market")
REF = "fhs_garch_t"
N_CHUNKS = 16


def _job(args: tuple[str, str, int, list[float], int, int]) -> tuple[str, str, int, int, dict[str, Any]]:
    pf, model, window, levels, lo, hi = args
    r = _RETURNS[pf]
    if model == "lgbm_quantile":
        cfg = load_config()["backtest"]
        v = lgbm_quantile_forecasts(
            r, window=window, levels=levels, start=lo, refit_every=cfg["ml_refit_every"]
        )
        res: dict[str, Any] = {"var": v, "es": {a: np.full(r.size - lo, np.nan) for a in levels}}
    elif model == "lgbm_vol_fhs":
        cfg = load_config()["backtest"]
        res = lgbm_vol_forecasts(r, window=window, levels=levels, start=lo, refit_every=cfg["ml_refit_every"])
    else:
        res = rolling_forecasts(r, model, window=window, levels=levels, start=lo, stop=hi)
    return pf, model, window, lo, res


_RETURNS: dict[str, np.ndarray[Any, Any]] = {}


def _init(returns: dict[str, np.ndarray[Any, Any]]) -> None:
    _RETURNS.update(returns)


def run_forecasts(pfs: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.DataFrame:
    levels = cfg["levels"]
    start = max(cfg["windows_sensitivity"])
    returns = {k: (v["portfolio"] * 100).to_numpy() for k, v in pfs.items()}
    n = next(iter(returns.values())).size
    bounds = np.linspace(start, n, N_CHUNKS + 1).astype(int)
    jobs: list[tuple[str, str, int, list[float], int, int]] = []
    for pf in pfs:
        windows = cfg["windows_sensitivity"] if pf == "main" else [cfg["window"]]
        for w in windows:
            for m in MODELS:
                jobs += [(pf, m, w, levels, int(lo), int(hi)) for lo, hi in itertools.pairwise(bounds)]
            # ML models: sequential refit schedule, one job each
            jobs += [(pf, m, w, levels, start, n) for m in ("lgbm_quantile", "lgbm_vol_fhs")]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_init, initargs=(returns,)) as ex:
        for pf, m, w, lo, res in ex.map(_job, jobs):
            dates = pfs[pf].index[lo : lo + len(res["var"][levels[0]])]
            for a in levels:
                rows.append(pd.DataFrame({"portfolio": pf, "model": m, "window": w, "level": a, "date": dates,
                                          "var": res["var"][a], "es": res["es"][a]}))  # fmt: skip
    print(f"forecasts: {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    f = pd.concat(rows, ignore_index=True)
    loss = pd.concat({k: -v["portfolio"] * 100 for k, v in pfs.items()}, names=["portfolio", "date"])
    f["loss"] = loss.reindex(pd.MultiIndex.from_frame(f[["portfolio", "date"]])).to_numpy()
    return f.sort_values(["portfolio", "model", "window", "level", "date"], ignore_index=True)


def _summ(g: pd.DataFrame, a: float) -> dict[str, Any]:
    es = g["es"].to_numpy() if g["es"].notna().all() else None
    return bt.summarize(g["loss"].to_numpy(), g["var"].to_numpy(), a, es)


def rolling_zones(g: pd.DataFrame) -> dict[str, float]:
    """Share of days whose trailing 250-day exception count falls in each Basel zone (99% VaR)."""
    hits = (g["loss"] > g["var"]).astype(int).rolling(250).sum().dropna().astype(int)
    z = hits.map(bt.traffic_light_zone)
    return {f"share_{k}": float((z == k).mean()) for k in ("green", "amber", "red")} | {
        "max_250d": int(hits.max())
    }


def evaluate(f: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    summary, per_year, regimes = [], [], []
    for key, g in f.groupby(["portfolio", "model", "window", "level"], sort=False):
        k4: tuple[Any, ...] = key
        pf, m, w, a = str(k4[0]), str(k4[1]), int(k4[2]), float(k4[3])
        s = {"portfolio": pf, "model": m, "window": w, "level": a, **_summ(g, a)}
        if a == 0.99:
            s |= rolling_zones(g)
        summary.append(s)
        if pf != "main":
            continue
        for y, gy in g.groupby(g["date"].dt.year):
            k = _summ(gy, a)
            per_year.append({"model": m, "window": w, "level": a, "year": y, "n": k["n"],
                             "exceptions": k["exceptions"], "p_uc": k["p_uc"], "p_cc": k["p_cc"],
                             "zone": bt.traffic_light_zone(k["exceptions"], k["n"], 1 - a)})  # fmt: skip
        for reg in cfg["regimes"]:
            gr = g[(g["date"] >= reg["start"]) & (g["date"] <= reg["end"])]
            k = _summ(gr, a)
            regimes.append({"model": m, "window": w, "level": a, "regime": reg["name"], "n": k["n"],
                            "exceptions": k["exceptions"], "expected": k["n"] * (1 - a), "p_uc": k["p_uc"],
                            "p_cc": k["p_cc"]})  # fmt: skip
    return {
        "summary": pd.DataFrame(summary),
        "per_year": pd.DataFrame(per_year),
        "regimes": pd.DataFrame(regimes),
    }


def comparisons(f: pd.DataFrame, window: int) -> dict[str, Any]:
    """DM vs the reference model and MCS (90%) on pinball (VaR) and FZ0 (VaR+ES), main portfolio."""
    m = f[(f["portfolio"] == "main") & (f["window"] == window)]
    out: dict[str, Any] = {"reference": REF, "window": window}
    for a in (0.99, 0.975):
        g = m[m["level"] == a]
        pin = g.assign(l=bt.pinball(g["loss"].to_numpy(), g["var"].to_numpy(), a)).pivot(
            index="date", columns="model", values="l")  # fmt: skip
        out[f"dm_pinball_{a}"] = {k: bt.diebold_mariano(pin[k].to_numpy(), pin[REF].to_numpy())
                                  for k in pin.columns if k != REF}  # fmt: skip
        out[f"mcs_pinball_{a}"] = _mcs(pin)
    g = m[(m["level"] == 0.975) & m["es"].notna()]
    fz = g.assign(l=bt.fz0(g["loss"].to_numpy(), g["var"].to_numpy(), g["es"].to_numpy(), 0.975)).pivot(
        index="date", columns="model", values="l")  # fmt: skip
    out["dm_fz0_0.975"] = {
        k: bt.diebold_mariano(fz[k].to_numpy(), fz[REF].to_numpy()) for k in fz.columns if k != REF
    }
    out["mcs_fz0_0.975"] = _mcs(fz)
    return out


def _mcs(losses: pd.DataFrame) -> dict[str, Any]:
    mcs = MCS(losses, size=0.10, reps=2000, block_size=20, method="R", seed=20261121)
    mcs.compute()
    return {
        "included": sorted(mcs.included),
        "pvalues": {str(k): float(v) for k, v in mcs.pvalues["Pvalue"].items()},
    }


def shuffled_label_real(pfs: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Leakage check on the real series: LightGBM trained on permuted targets vs the real fit vs HS."""
    r = (pfs["main"]["portfolio"] * 100).to_numpy()
    w, a, start = cfg["window"], 0.99, max(cfg["windows_sensitivity"])
    loss = -r[start:]
    out: dict[str, Any] = {"level": a, "window": w}
    for name, shuffle in (("lgbm_real", False), ("lgbm_shuffled", True)):
        v = lgbm_quantile_forecasts(r, window=w, levels=[a], start=start, refit_every=cfg["ml_refit_every"],
                                    shuffle_labels=shuffle)[a]  # fmt: skip
        out[name] = {"pinball": float(bt.pinball(loss, v, a).mean()), "rate": float((loss > v).mean())}
    hs = rolling_forecasts(r, "historical", window=w, levels=[a], start=start)["var"][a]
    out["historical"] = {"pinball": float(bt.pinball(loss, hs, a).mean()), "rate": float((loss > hs).mean())}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    cfg_all = load_config()
    cfg = cfg_all["backtest"]
    pfs = {"main": load_portfolio("weights"), "equal": load_portfolio("weights_equal")}
    f = run_forecasts(pfs, cfg)
    f.to_parquet(ART / "forecasts.parquet")
    ev = evaluate(f, cfg)
    for k, v in ev.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    comp = comparisons(f, cfg["window"])
    p = pfs["main"]
    meta = {
        "run": {"script": "scripts/r3_backtest.py", "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "arch": arch.__version__},
        "config": cfg_all,
        "data": {"first_return": str(p.index[0].date()), "last_return": str(p.index[-1].date()),
                 "n_returns": len(p), "first_forecast": str(f["date"].min().date()),
                 "n_forecast_days": int(((f["portfolio"] == "main") & (f["model"] == REF)
                                         & (f["window"] == cfg["window"]) & (f["level"] == 0.99)).sum()),
                 "portfolio_daily_sd_pct": float(p["portfolio"].std() * 100),
                 "asset_daily_sd_pct": {c: float(p[c].std() * 100) for c in p.columns if c != "portfolio"},
                 "worst_day": str(p.index[int(np.argmin(p["portfolio"].to_numpy()))].date()),
                 "worst_day_loss_pct": float(-p["portfolio"].min() * 100)},
        "zone_counts_250_99": bt.zone_counts(250, 0.01),
        "zone_counts_250_975": bt.zone_counts(250, 0.025),
        "comparisons": comp,
        "shuffled_label_real": shuffled_label_real(pfs, cfg),
    }  # fmt: skip
    (OUT / "backtest.json").write_text(json.dumps(meta, indent=2, default=str))
    s = ev["summary"]
    print(s[(s["portfolio"] == "main") & (s["window"] == cfg["window"])].round(4).to_string())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
