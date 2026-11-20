"""Loaders for credit datasets. Raw files live in data/raw/ (git-ignored, see docs/data.md)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GMSC_DIR = Path("data/raw/givemesomecredit")
GMSC_TARGET = "default_2y"

# Kaggle column -> snake_case. Meanings from the competition's Data Dictionary.xls.
GMSC_COLUMNS = {
    "SeriousDlqin2yrs": GMSC_TARGET,  # 90+ days past due within 2 years
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "n_30_59_dpd",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "n_open_lines",
    "NumberOfTimes90DaysLate": "n_90_dpd",
    "NumberRealEstateLoansOrLines": "n_real_estate",
    "NumberOfTime60-89DaysPastDueNotWorse": "n_60_89_dpd",
    "NumberOfDependents": "n_dependents",
}
GMSC_FEATURES = [c for c in GMSC_COLUMNS.values() if c != GMSC_TARGET]
# Past-due counts contain 96 and 98, which are codes, not counts (269 rows in cs-training).
GMSC_SPECIAL_CODES = {c: [96, 98] for c in ("n_30_59_dpd", "n_60_89_dpd", "n_90_dpd")}


def load_gmsc_train(root: Path = GMSC_DIR) -> pd.DataFrame:
    """Labelled training file only (cs-test.csv has no labels)."""
    df = pd.read_csv(root / "cs-training.csv", index_col=0)
    df = df.rename(columns=GMSC_COLUMNS)
    missing = set(GMSC_COLUMNS.values()) - set(df.columns)
    if missing:
        raise ValueError(f"unexpected schema, missing {sorted(missing)}")
    return df[[GMSC_TARGET, *GMSC_FEATURES]]


def load_gmsc_test(root: Path = GMSC_DIR) -> pd.DataFrame:
    """Kaggle test file: same features, target column empty. Index = Kaggle `Id`."""
    df = pd.read_csv(root / "cs-test.csv", index_col=0).rename(columns=GMSC_COLUMNS)
    df.index.name = "Id"
    if df[GMSC_TARGET].notna().any():
        raise ValueError("cs-test.csv unexpectedly contains labels")
    return df[GMSC_FEATURES]
