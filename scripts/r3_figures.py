"""R3 figures: 99% VaR forecasts vs realised losses (three models) and passed years per model.

Palette: dataviz reference categorical slots in fixed order (blue, orange, aqua), light surface; losses in
muted ink. One axis per chart, thin lines, recessive grid, legend + direct labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from r2_figures import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style

ART = Path("data/processed/market/forecasts.parquet")
OUT = Path("results/market/backtest")
FIG = Path("reports/figures")
AQUA = "#1baf9b"
SHOW = [
    ("historical", ORANGE, "Historical simulation"),
    ("normal", AQUA, "Normal"),
    ("fhs_garch_t", BLUE, "FHS GARCH-t"),
]


def var_path() -> None:
    f = pd.read_parquet(ART)
    f = f[(f["portfolio"] == "main") & (f["window"] == 1000) & (f["level"] == 0.99)]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax)
    base = f[f["model"] == "fhs_garch_t"].set_index("date")
    ax.bar(
        base.index,
        base["loss"].clip(lower=0),
        width=1.5,
        color=GRID,
        label="daily loss (gains not shown)",
        zorder=1,
    )
    for m, c, label in SHOW:
        g = f[f["model"] == m].set_index("date")
        ax.plot(g.index, g["var"], color=c, linewidth=1.2, label=f"99% VaR, {label}", zorder=3)
    hit = base[base["loss"] > base["var"]]
    ax.scatter(hit.index, hit["loss"], s=14, color=INK, zorder=4, label="exception (FHS GARCH-t)")
    ax.set_ylabel("Loss, % of portfolio value", color=INK, fontsize=10)
    ax.set_title("One-day 99% VaR vs realised loss, 1000-day window (ECB FX/rates portfolio)",
                 color=INK, fontsize=11, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(FIG / "market_var_path.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def passed_years() -> None:
    py = pd.read_csv(OUT / "per_year.csv")
    m = py[(py["window"] == 1000) & (py["level"] == 0.99)]
    m = m.assign(passed=(m["p_uc"] > 0.05) & (m["p_cc"] > 0.05))
    t = m.groupby("model").agg(passed=("passed", "sum"), years=("year", "nunique")).sort_values("passed")
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(t.index, t["passed"], height=0.6, color=BLUE, zorder=3)
    for i, (v, n) in enumerate(zip(t["passed"], t["years"], strict=True)):
        ax.text(v + 0.2, i, f"{v} of {n}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Calendar years passing Kupiec and Christoffersen CC (5%)", color=INK, fontsize=10)
    ax.set_title("99% VaR: years passed, 1000-day window", color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, t["years"].max() + 2)
    fig.tight_layout()
    fig.savefig(FIG / "market_passed_years.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    var_path()
    passed_years()
