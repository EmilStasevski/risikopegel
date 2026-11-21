"""R4 (Module C): rolling-origin forecasts for 15 monthly macro series, 8 models, with compute and energy.

Protocol (fixed before the run):
- origins: every month from 2016-01 to the month before each series ends; at least 48 months of history
- horizons 1..12 forecast; scored at h = 1, 3, 12 (the LightGBM direct models exist only for these)
- every forecast uses the history up to its origin only; classical specs re-selected at January origins
- metrics: MASE (scale: in-sample seasonal-naive MAE, m = 12, up to the origin), sMAPE, scaled pinball loss
  over the 0.1/0.5/0.9 quantiles, 80% interval coverage; DM test vs seasonal naive on scaled absolute errors
- all models are scored on the same (series, origin, h) cells
- compute: wall time and CPU-package energy (RAPL) per model block, machine otherwise idle

Data caveat: Eurostat publishes revised values, not real-time vintages (docs/data.md).
Writes results/macro/forecast/*; forecasts to data/processed/macro/forecasts.parquet.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from risikopegel.forecast import models as fm_cls
from risikopegel.forecast.data import is_seasonal, load_panel
from risikopegel.forecast.energy import idle_power, measure
from risikopegel.market.backtest import diebold_mariano

OUT = Path("results/macro/forecast")
ART = Path("data/processed/macro")
FIRST_ORIGIN = pd.Period("2016-01", freq="M")
MIN_TRAIN = 48
HS = (1, 3, 12)
H = 12
CLASSICAL = ("naive", "seasonal_naive", "ets", "arima")
FOUNDATION = ("chronos2", "chronos_bolt_small", "timesfm25")


def origins_for(s: pd.Series) -> list[pd.Period]:
    first = max(FIRST_ORIGIN, s.index[0] + MIN_TRAIN - 1)
    return list(pd.period_range(first, s.index[-1] - 1, freq="M"))


def _classical_series(args: tuple[str, str, pd.Series]) -> list[dict[str, Any]]:
    # one BLAS thread per worker: 15 workers x multithreaded BLAS oversubscribed the CPU 3x in the first run
    with threadpool_limits(limits=1):
        return _classical_series_inner(args)


def _classical_series_inner(args: tuple[str, str, pd.Series]) -> list[dict[str, Any]]:
    model, sid, s = args
    cache = fm_cls.SpecCache()
    rows = []
    for o in origins_for(s):
        y = s[:o].to_numpy()
        if model == "naive":
            q = fm_cls.naive(y)
        elif model == "seasonal_naive":
            q = fm_cls.seasonal_naive(y)
        elif model == "ets":
            q = fm_cls.ets(y, cache, sid, reselect=o.month == 1)
        else:
            q = fm_cls.arima(y, cache, sid, reselect=o.month == 1, seasonal=is_seasonal(sid))
        rows += [
            {
                "series": sid,
                "origin": str(o),
                "h": h,
                "q10": q[h - 1, 0],
                "q50": q[h - 1, 1],
                "q90": q[h - 1, 2],
            }
            for h in HS
        ]
    return rows


def run_classical(model: str, panel: dict[str, pd.Series]) -> list[dict[str, Any]]:
    jobs = [(model, sid, s) for sid, s in panel.items()]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
        return [r for rows in ex.map(_classical_series, jobs) for r in rows]


def run_lgbm(panel: dict[str, pd.Series]) -> list[dict[str, Any]]:
    all_origins = sorted({o for s in panel.values() for o in origins_for(s)})
    rows: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    for o in all_origins:
        hist = {sid: s[:o] for sid, s in panel.items() if s.index[0] <= o}
        data = {sid: (h.to_numpy(), pd.PeriodIndex(h.index).month.to_numpy()) for sid, h in hist.items()}
        if o.month == 1 or not models:
            models = {h: fm_cls.lgb_fit(data, h) for h in HS}
        for sid, s in panel.items():
            if o not in origins_for(s):
                continue
            y, month = data[sid]
            for h in HS:
                q = fm_cls.lgb_predict(models[h], y, month)
                rows.append({"series": sid, "origin": str(o), "h": h, "q10": q[0], "q50": q[1], "q90": q[2]})
    return rows


def run_foundation(name: str, panel: dict[str, pd.Series]) -> list[dict[str, Any]]:
    from risikopegel.forecast.fm import load

    predict = load(name, H)
    rows: list[dict[str, Any]] = []
    all_origins = sorted({o for s in panel.values() for o in origins_for(s)})
    for o in all_origins:
        ids = [sid for sid, s in panel.items() if o in origins_for(s)]
        q = predict([panel[sid][:o].to_numpy() for sid in ids])
        for i, sid in enumerate(ids):
            rows += [
                {
                    "series": sid,
                    "origin": str(o),
                    "h": h,
                    "q10": q[i, h - 1, 0],
                    "q50": q[i, h - 1, 1],
                    "q90": q[i, h - 1, 2],
                }
                for h in HS
            ]
    return rows


def attach_truth(f: pd.DataFrame, panel: dict[str, pd.Series]) -> pd.DataFrame:
    actual, scale = [], []
    sc_cache: dict[tuple[str, str], float] = {}
    for sid, o, h in zip(f["series"], f["origin"], f["h"], strict=True):
        s, op = panel[sid], pd.Period(o, freq="M")
        t = op + h
        actual.append(float(s[t]) if t in s.index else np.nan)
        if (sid, o) not in sc_cache:
            y = s[:op].to_numpy()
            sc_cache[(sid, o)] = float(np.mean(np.abs(y[12:] - y[:-12])))
        scale.append(sc_cache[(sid, o)])
    return f.assign(actual=actual, scale=scale)


def score(f: pd.DataFrame) -> pd.DataFrame:
    e = f["actual"] - f["q50"]
    f = f.assign(
        ase=np.abs(e) / f["scale"],
        smape=200 * np.abs(e) / (np.abs(f["actual"]) + np.abs(f["q50"])),
        covered=((f["actual"] >= f["q10"]) & (f["actual"] <= f["q90"])).astype(float),
    )
    pin: Any = 0.0
    for q, c in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        d = f["actual"] - f[c]
        pin = pin + np.maximum(q * d, (q - 1) * d)
    return f.assign(spl=(pin / 3) / f["scale"])


def evaluate(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    f = f.dropna(subset=["actual"])
    n_models = f["model"].nunique()
    cell = f.groupby(["series", "origin", "h"])["model"].transform("nunique")
    f = score(f[cell == n_models])
    by_h = (
        f.groupby(["model", "h"])
        .agg(
            n=("ase", "size"),
            mase=("ase", "mean"),
            smape=("smape", "mean"),
            spl=("spl", "mean"),
            coverage80=("covered", "mean"),
        )
        .reset_index()
    )
    by_series = (
        f.groupby(["model", "series", "h"]).agg(mase=("ase", "mean"), spl=("spl", "mean")).reset_index()
    )
    dm_rows = []
    for ref in ("seasonal_naive", "ets"):
        dm_rows += _dm_rows(f, ref)
    return {"by_h": by_h, "by_series": by_series, "dm": pd.DataFrame(dm_rows)}


def _dm_rows(f: pd.DataFrame, ref: str) -> list[dict[str, Any]]:
    dm_rows = []
    base = f[f["model"] == ref].set_index(["series", "origin", "h"])["ase"]
    for m in f["model"].unique():
        if m == ref:
            continue
        cur = f[f["model"] == m].set_index(["series", "origin", "h"])["ase"]
        for h in HS:
            # pooled: average scaled error across series per origin, then DM over origins (HAC lag >= h-1)
            a = cur.xs(h, level="h").groupby(level="origin").mean()
            b = base.xs(h, level="h").groupby(level="origin").mean()
            dm = diebold_mariano(a.to_numpy(), b.reindex(a.index).to_numpy(), lag=max(h - 1, 4))
            per_series = []
            for sid in f["series"].unique():
                x = cur.xs((sid, h), level=("series", "h"))
                y = base.xs((sid, h), level=("series", "h")).reindex(x.index)
                per_series.append(diebold_mariano(x.to_numpy(), y.to_numpy(), lag=max(h - 1, 4)))
            dm_rows.append(
                {
                    "reference": ref,
                    "model": m,
                    "h": h,
                    "pooled_stat": dm["stat"],
                    "pooled_p": dm["p_two_sided"],
                    "series_better_sig": sum(d["stat"] < 0 and d["p_two_sided"] < 0.05 for d in per_series),
                    "series_worse_sig": sum(d["stat"] > 0 and d["p_two_sided"] < 0.05 for d in per_series),
                    "n_series": len(per_series),
                }
            )
    return dm_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    if "--evaluate-only" in sys.argv:  # re-score stored forecasts (no new model runs)
        for k, v in evaluate(pd.read_parquet(ART / "forecasts.parquet")).items():
            v.to_csv(OUT / f"{k}.csv", index=False)
        return
    p_idle = idle_power()
    runners: dict[str, Callable[[], list[dict[str, Any]]]] = {
        m: functools.partial(run_classical, m, panel) for m in CLASSICAL
    }
    runners["lgbm_global"] = functools.partial(run_lgbm, panel)
    for name in FOUNDATION:
        runners[name] = functools.partial(run_foundation, name, panel)
    rows, compute = [], {}
    for m, fn in runners.items():
        with measure() as meas:
            r = fn()
        rows += [{**x, "model": m} for x in r]
        n_fc = len({(x["series"], x["origin"]) for x in r})
        compute[m] = {
            "seconds": meas.seconds,
            "joules": meas.joules,
            "n_forecasts": n_fc,
            "ms_per_forecast": 1000 * meas.seconds / n_fc,
            "joules_per_forecast": None if meas.joules is None else meas.joules / n_fc,
            "joules_above_idle_per_forecast": None
            if meas.joules is None or p_idle is None
            else (meas.joules - p_idle * meas.seconds) / n_fc,
        }
        print(m, json.dumps(compute[m]), flush=True)
    f = attach_truth(pd.DataFrame(rows), panel)
    f.to_parquet(ART / "forecasts.parquet")
    ev = evaluate(f)
    for k, v in ev.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    meta = {
        "run": {
            "script": "scripts/r4_forecast.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "cpu_count": os.cpu_count(),
            "idle_package_watts": p_idle,
            "energy_source": "Intel RAPL package-0 (whole CPU package), CPU only, no GPU",
        },
        "series": {
            sid: {
                "first": str(s.index[0]),
                "last": str(s.index[-1]),
                "n": len(s),
                "n_origins": len(origins_for(s)),
                "seasonal_nsa": is_seasonal(sid),
            }
            for sid, s in panel.items()
        },
        "protocol": {"first_origin": str(FIRST_ORIGIN), "min_train": MIN_TRAIN, "horizons": list(HS)},
        "compute": compute,
    }
    (OUT / "forecast.json").write_text(json.dumps(meta, indent=2))
    print(ev["by_h"].pivot(index="model", columns="h", values="mase").round(3))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
