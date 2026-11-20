"""Home Credit modelling table (from the dbt mart) and feature roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DB = Path("data/processed/risikopegel.duckdb")
TARGET = "target"
ID = "sk_id_curr"
# Kept out of the model on purpose: used only for the fairness check (EU/German anti-discrimination law;
# see docs/model_cards). Not a claim of legal compliance.
PROTECTED = ["gender"]
CATEGORICAL = [
    "contract_type",
    "income_type",
    "education",
    "family_status",
    "housing_type",
    "occupation",
    "bureau_history_status",
]
BOOLEAN = ["owns_car", "owns_realty", "employed_placeholder"]
DROP = [ID, TARGET, "is_train", *PROTECTED]


@dataclass(frozen=True)
class HCData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    X_kaggle: pd.DataFrame  # unlabelled Kaggle test = drift scenario
    protected_test: pd.DataFrame
    numeric: list[str]
    categorical: list[str]


def load_mart(db: Path = DB) -> pd.DataFrame:
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute("select * from mart_hc__applicant_features order by sk_id_curr").df()


def prepare(df: pd.DataFrame, *, seed: int, test_size: float = 0.2, valid_size: float = 0.2) -> HCData:
    """Stratified split (no time proxy exists, see docs/split_policy.md): train / valid / test.

    valid_size is the share of the non-test part (default 0.2 -> 64/16/20 overall).
    """
    df = df.copy()
    for c in BOOLEAN:
        df[c] = df[c].astype(float)
    for c in CATEGORICAL:
        df[c] = df[c].astype("string").fillna("MISSING").astype("category")
    features = [c for c in df.columns if c not in DROP]
    numeric = [c for c in features if c not in CATEGORICAL]
    df[numeric] = df[numeric].astype(float)

    lab = df[df["is_train"]].reset_index(drop=True)
    kag = df[~df["is_train"]].reset_index(drop=True)
    y = lab[TARGET].astype(int).to_numpy()
    idx = np.arange(len(lab))
    i_rest, i_test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    i_tr, i_va = train_test_split(i_rest, test_size=valid_size, stratify=y[i_rest], random_state=seed)
    X = lab[features]
    return HCData(
        X_train=X.iloc[i_tr].reset_index(drop=True),
        y_train=y[i_tr],
        X_valid=X.iloc[i_va].reset_index(drop=True),
        y_valid=y[i_va],
        X_test=X.iloc[i_test].reset_index(drop=True),
        y_test=y[i_test],
        X_kaggle=kag[features],
        protected_test=lab.iloc[i_test][PROTECTED].reset_index(drop=True),
        numeric=numeric,
        categorical=list(CATEGORICAL),
    )
