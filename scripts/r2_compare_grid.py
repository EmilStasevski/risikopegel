"""R2: tuned (Optuna) vs grid models on the same test set, per family, paired bootstrap.

Needs both prediction sets: data/processed/hc_challengers_grid/ (grid run, kept as the baseline) and
data/processed/hc_challengers/ (tuned run). Writes results/credit/hc_tuning/tuned_vs_grid.json.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from r2_challengers import SEED, paired_bootstrap

GRID = Path("data/processed/hc_challengers_grid/test_predictions.parquet")
TUNED = Path("data/processed/hc_challengers/test_predictions.parquet")
RES_GRID = Path("results/credit/hc_challengers_grid/results.json")
RES_TUNED = Path("results/credit/hc_challengers/results.json")
OUT = Path("results/credit/hc_tuning/tuned_vs_grid.json")


def main() -> None:
    g, t = pd.read_parquet(GRID), pd.read_parquet(TUNED)
    if not np.array_equal(g["y"].to_numpy(), t["y"].to_numpy()):
        raise SystemExit("test labels differ between runs: splits are not identical")
    rg, rt = json.loads(RES_GRID.read_text()), json.loads(RES_TUNED.read_text())
    y = t["y"].to_numpy()
    rng = np.random.default_rng(SEED)
    fam: dict[str, Any] = {}
    for k in rt["models"]:
        a, b = rg["models"][k], rt["models"][k]
        fam[k] = {
            "grid": {
                "valid_auc": a["valid_auc"],
                "test_auc": a["test"]["auc"],
                "params": a["selected_params"],
            },
            "tuned": {
                "valid_auc": b["valid_auc"],
                "test_auc": b["test"]["auc"],
                "params": b["selected_params"],
            },
            "valid_gain": b["valid_auc"] - a["valid_auc"],
            "test_tuned_minus_grid": paired_bootstrap(y, g[k].to_numpy(), t[k].to_numpy(), rng),
            "tuned_valid_minus_test": b["valid_auc"] - b["test"]["auc"],
            "grid_valid_minus_test": a["valid_auc"] - a["test"]["auc"],
        }
    out = {
        "run": {
            "script": "scripts/r2_compare_grid.py",
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "champion": {"grid": rg["champion"]["name"], "tuned": rt["champion"]["name"]},
        "families": fam,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in fam.items():
        c = v["test_tuned_minus_grid"]
        print(
            f"{k:18s} valid {v['valid_gain']:+.4f}  test {c['auc_diff']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]"
        )


if __name__ == "__main__":
    main()
