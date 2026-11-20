"""Build a Kaggle late-submission file for Give Me Some Credit from the smoke-test scorecard.

The final model is fitted on all 150,000 labelled rows with the same config as the smoke test (no tuning).
Output: submissions/gmsc_woe_scorecard.csv (Id,Probability), manifest in results/credit/gmsc_submission/.
The file is uploaded by the user, not by this script.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from risikopegel.credit.data import (
    GMSC_FEATURES,
    GMSC_SPECIAL_CODES,
    GMSC_TARGET,
    load_gmsc_test,
    load_gmsc_train,
)
from risikopegel.credit.scorecard import ScorecardConfig, fit_predict_pd

SUB = Path("submissions/gmsc_woe_scorecard.csv")
OUT = Path("results/credit/gmsc_submission")


def main() -> None:
    train, test = load_gmsc_train(), load_gmsc_test()
    cfg = ScorecardConfig(special_codes=GMSC_SPECIAL_CODES)
    _, p = fit_predict_pd(GMSC_FEATURES, cfg, train, train[GMSC_TARGET].to_numpy(), test)
    if not (np.isfinite(p).all() and ((p > 0) & (p < 1)).all()):
        raise ValueError("predictions outside (0, 1)")
    sub = pd.DataFrame({"Id": test.index.astype(int), "Probability": p})
    sample = pd.read_csv("data/raw/givemesomecredit/sampleEntry.csv")
    if len(sub) != len(sample) or not (sub["Id"].to_numpy() == sample["Id"].to_numpy()).all():
        raise ValueError("Ids do not match sampleEntry.csv")
    SUB.parent.mkdir(exist_ok=True)
    sub.to_csv(SUB, index=False, float_format="%.9f")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "file": str(SUB),
        "sha256": hashlib.sha256(SUB.read_bytes()).hexdigest(),
        "rows": len(sub),
        "model": "WoE/logistic scorecard, smoke-test config, fitted on all 150,000 cs-training rows",
        "mean_probability": float(p.mean()),
        "kaggle_score": None,  # filled in from Kaggle after the user uploads
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
