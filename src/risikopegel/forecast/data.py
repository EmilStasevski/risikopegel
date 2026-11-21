"""Monthly macro panel from the raw ECB/Eurostat files (catalog entries with module = "macro")."""

from __future__ import annotations

import pandas as pd

from risikopegel.sources import load_catalog, load_raw

NSA = ("hicp_",)  # not seasonally adjusted (all other macro series are SA or SCA)


def load_panel() -> dict[str, pd.Series]:
    out = {}
    for s in load_catalog():
        if s.module != "macro":
            continue
        df = load_raw(s).dropna()
        out[s.id] = pd.Series(df["value"].to_numpy(dtype=float), index=pd.PeriodIndex(df["period"], freq="M"))
    return out


def is_seasonal(series_id: str) -> bool:
    return series_id.startswith(NSA)
