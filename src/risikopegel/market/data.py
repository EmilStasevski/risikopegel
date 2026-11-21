"""Daily portfolio returns from ECB reference FX rates and the AAA government yield curve."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risikopegel.sources import load_catalog, load_raw

CONFIG = Path("config/market.toml")
FX = ["fx_usd", "fx_gbp", "fx_jpy", "fx_chf"]
BONDS = {"bond_2y": ("yc_aaa_2y", 2.0), "bond_10y": ("yc_aaa_10y", 10.0)}
DAY = 1.0 / 252.0


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def asset_log_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """levels: fx_* (foreign per EUR) and yc_* (spot %, continuous comp.) on common dates.

    Returns daily log returns in EUR of each asset; row t uses levels at t-1 and t only.
    """
    out = pd.DataFrame(index=levels.index)
    for c in FX:
        out[c] = -np.log(levels[c]).diff()  # EUR value of 1 foreign unit = 1 / rate
    for name, (yc, mat) in BONDS.items():
        y = levels[yc] / 100.0
        out[name] = -mat * y.diff() + y.shift(1) * DAY
    return out.iloc[1:]


def portfolio_returns(asset_lr: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Simple return of a daily-rebalanced fixed-weight portfolio."""
    w = pd.Series(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    simple = asset_lr[list(w.index)].apply(np.expm1)
    out: pd.Series = (simple * w).sum(axis=1)
    return out.rename("portfolio")


def load_levels(start: str) -> pd.DataFrame:
    cat = {s.id: s for s in load_catalog()}
    cols = {}
    for sid in [*FX, *(yc for yc, _ in BONDS.values())]:
        df = load_raw(cat[sid])
        cols[sid] = pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["period"]))
    lv = pd.DataFrame(cols)
    # common business days on which every series has a published value (no forward fill: a stale price
    # would show up as a zero return and then a jump)
    return lv[lv.index >= pd.Timestamp(start)].dropna()


def load_portfolio(weights_key: str = "weights") -> pd.DataFrame:
    cfg = load_config()
    lr = asset_log_returns(load_levels(cfg["portfolio"]["start"]))
    lr["portfolio"] = portfolio_returns(lr, cfg["portfolio"][weights_key])
    return lr
