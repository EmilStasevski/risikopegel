"""R2 figures (static PNG for the Markdown reports), drawn from results/ and the run artifacts.

Palette: validated categorical slots 1-2 (blue = champion, orange = scorecard) from the dataviz reference
palette, light surface. One axis per chart, thin marks, recessive grid, legend + direct labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("results/credit/hc_challengers/results.json")
EXP = Path("results/credit/hc_explain/global_importance.csv")
ART = Path("data/processed/hc_challengers")
FIG = Path("reports/figures")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def calibration(r: dict[str, Any], champ: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    _style(ax)
    lim = 0.0
    for name, color in ((champ, BLUE), ("scorecard", ORANGE)):
        t = pd.DataFrame(r["models"][name]["calibration_test"])
        lim = max(lim, t["mean_pd"].max(), t["default_rate"].max())
        ax.plot(t["mean_pd"], t["default_rate"], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)  # fmt: skip
        last = t.iloc[-1]
        ax.annotate(name, (last["mean_pd"], last["default_rate"]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")  # fmt: skip
    lim *= 1.08
    ax.plot(
        [0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--", label="perfect calibration", zorder=2
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted PD per decile", color=INK, fontsize=10)
    ax.set_ylabel("Observed default rate", color=INK, fontsize=10)
    ax.set_title("Calibration on test (equal-frequency deciles)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "hc_calibration.png", dpi=150)
    plt.close(fig)


def score_shift(champ: str) -> None:
    va = pd.read_parquet(ART / "valid_predictions.parquet")[champ].to_numpy()
    kg = pd.read_parquet(ART / "kaggle_predictions.parquet")[champ].to_numpy()
    bins = np.linspace(0, np.quantile(np.r_[va, kg], 0.995), 41)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _style(ax)
    for x, color, lab in ((va, BLUE, "validation (train population)"), (kg, ORANGE, "Kaggle test (shifted)")):
        h, _ = np.histogram(x, bins=bins)
        ax.step(bins[:-1], h / h.sum(), where="post", color=color, linewidth=2, label=lab)
    ax.set_xlabel(f"Predicted PD ({champ})", color=INK, fontsize=10)
    ax.set_ylabel("Share of applicants", color=INK, fontsize=10)
    ax.set_title("Score distribution: train population vs drift scenario", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "hc_score_shift.png", dpi=150)
    plt.close(fig)


def importance(champ: str) -> None:
    imp = pd.read_csv(EXP, index_col="feature")
    top = imp[champ].sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(top.index, top.to_numpy(), color=BLUE, height=0.6)
    for i, v in enumerate(top.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("Mean |contribution| (log-odds)", color=INK, fontsize=10)
    ax.set_title(f"Global importance, {champ} (top 12 features)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "hc_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads(RES.read_text())
    champ = r["champion"]["name"]
    calibration(r, champ)
    score_shift(champ)
    importance(champ)
    print("figures:", sorted(p.name for p in FIG.glob("hc_*.png")))


if __name__ == "__main__":
    main()
