"""Synthetic raw CSVs with the Kaggle schemas (only the columns the dbt models read).

Generated at test time; no real data is committed (licenses forbid redistribution).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HC_APP_COLS = [
    "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_YEAR",
]  # fmt: skip


def _applications(rng: np.random.Generator, ids: np.ndarray, with_target: bool) -> pd.DataFrame:
    n = ids.size
    df = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "CODE_GENDER": rng.choice(["F", "M"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "CNT_CHILDREN": rng.integers(0, 3, n),
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n).round(1),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, n).round(1),
            "AMT_ANNUITY": rng.uniform(5e3, 5e4, n).round(1),
            "AMT_GOODS_PRICE": rng.uniform(1e5, 1e6, n).round(1),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], n),
            "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"], n),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n),
            "NAME_HOUSING_TYPE": "House / apartment",
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n).round(6),
            "DAYS_BIRTH": -rng.integers(7000, 25000, n),
            "DAYS_EMPLOYED": np.where(rng.random(n) < 0.2, 365243, -rng.integers(0, 15000, n)),
            "DAYS_REGISTRATION": -rng.integers(0, 20000, n).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
            "OWN_CAR_AGE": np.where(rng.random(n) < 0.6, np.nan, rng.integers(0, 30, n)),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", ""], n),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n),
            "EXT_SOURCE_1": np.where(rng.random(n) < 0.5, np.nan, rng.random(n).round(6)),
            "EXT_SOURCE_2": rng.random(n).round(6),
            "EXT_SOURCE_3": rng.random(n).round(6),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, n).astype(float),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 5, n).astype(float),
        }
    )
    if with_target:
        df.insert(1, "TARGET", (rng.random(n) < 0.1).astype(int))
    return df


def write_fixtures(root: Path, *, seed: int = 0, inject_future_payment: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    gm, hc = root / "givemesomecredit", root / "homecredit"
    gm.mkdir(parents=True)
    hc.mkdir(parents=True)

    n = 200
    gmsc = pd.DataFrame(
        {
            "SeriousDlqin2yrs": (rng.random(n) < 0.07).astype(int),
            "RevolvingUtilizationOfUnsecuredLines": rng.random(n),
            "age": rng.integers(21, 90, n),
            "NumberOfTime30-59DaysPastDueNotWorse": np.where(np.arange(n) < 3, 98, rng.integers(0, 3, n)),
            "DebtRatio": rng.random(n),
            "MonthlyIncome": np.where(rng.random(n) < 0.2, np.nan, rng.integers(1000, 15000, n)),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
            "NumberOfTimes90DaysLate": np.where(np.arange(n) < 3, 98, rng.integers(0, 2, n)),
            "NumberRealEstateLoansOrLines": rng.integers(0, 4, n),
            "NumberOfTime60-89DaysPastDueNotWorse": np.where(np.arange(n) < 3, 98, rng.integers(0, 2, n)),
            "NumberOfDependents": np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 4, n)),
        },
        index=pd.RangeIndex(1, n + 1),
    )
    gmsc.to_csv(gm / "cs-training.csv", na_rep="NA")  # GMSC writes missing values as NA

    train_ids, test_ids = np.arange(100001, 100151), np.arange(200001, 200031)
    _applications(rng, train_ids, True).to_csv(hc / "application_train.csv", index=False)
    _applications(rng, test_ids, False).to_csv(hc / "application_test.csv", index=False)
    all_ids = np.concatenate([train_ids, test_ids])

    nb = 300
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": rng.choice(all_ids, nb),
            "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + nb),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], nb),
            "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card"], nb),
            "DAYS_CREDIT": -rng.integers(1, 3000, nb),
            "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 30], nb),
            "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 2000, nb).astype(float),
            "DAYS_ENDDATE_FACT": np.where(rng.random(nb) < 0.5, np.nan, -rng.integers(1, 2000, nb)),
            "AMT_CREDIT_MAX_OVERDUE": np.nan,
            "CNT_CREDIT_PROLONG": 0,
            "AMT_CREDIT_SUM": rng.uniform(1e4, 1e6, nb).round(1),
            "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 1e5, nb).round(1),
            "AMT_CREDIT_SUM_OVERDUE": 0.0,
            # two rows updated after the application: must be dropped by staging
            "DAYS_CREDIT_UPDATE": np.where(np.arange(nb) < 2, 5, -rng.integers(0, 500, nb)),
        }
    )
    bureau.to_csv(hc / "bureau.csv", index=False)
    pd.DataFrame(
        {
            "SK_ID_BUREAU": np.repeat(bureau["SK_ID_BUREAU"].to_numpy()[:50], 4),
            "MONTHS_BALANCE": np.tile([0, -1, -2, -13], 50),  # month 0 must be dropped by staging
            "STATUS": rng.choice(["C", "0", "1", "X"], 200),
        }
    ).to_csv(hc / "bureau_balance.csv", index=False)

    npv = 120
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(1_000_000, 1_000_000 + npv),
            "SK_ID_CURR": rng.choice(all_ids, npv),
            "NAME_CONTRACT_TYPE": "Consumer loans",
            "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], npv),
            "AMT_APPLICATION": rng.uniform(1e4, 5e5, npv).round(1),
            "AMT_CREDIT": rng.uniform(1e4, 5e5, npv).round(1),
            "DAYS_DECISION": -rng.integers(1, 2900, npv),
        }
    )
    prev.to_csv(hc / "previous_application.csv", index=False)

    ni = 400
    pick = rng.integers(0, npv, ni)
    due = -rng.integers(1, 2000, ni).astype(float)
    paid = due + rng.integers(-10, 20, ni)
    paid = np.minimum(paid, -1.0)
    paid[rng.random(ni) < 0.03] = np.nan
    if inject_future_payment:
        paid[0] = 15.0  # a payment recorded after the application date
    pd.DataFrame(
        {
            "SK_ID_PREV": prev["SK_ID_PREV"].to_numpy()[pick],
            "SK_ID_CURR": prev["SK_ID_CURR"].to_numpy()[pick],
            "NUM_INSTALMENT_VERSION": 1.0,
            "NUM_INSTALMENT_NUMBER": 1,
            "DAYS_INSTALMENT": due,
            "DAYS_ENTRY_PAYMENT": paid,
            "AMT_INSTALMENT": 1000.0,
            "AMT_PAYMENT": rng.choice([1000.0, 900.0], ni),
        }
    ).to_csv(hc / "installments_payments.csv", index=False)

    nm = 200
    pick = rng.integers(0, npv, nm)
    pd.DataFrame(
        {
            "SK_ID_PREV": prev["SK_ID_PREV"].to_numpy()[pick],
            "SK_ID_CURR": prev["SK_ID_CURR"].to_numpy()[pick],
            "MONTHS_BALANCE": -rng.integers(1, 96, nm),
            "SK_DPD": rng.choice([0, 0, 5], nm),
            "SK_DPD_DEF": 0,
        }
    ).to_csv(hc / "POS_CASH_balance.csv", index=False)
    pd.DataFrame(
        {
            "SK_ID_PREV": prev["SK_ID_PREV"].to_numpy()[pick],
            "SK_ID_CURR": prev["SK_ID_CURR"].to_numpy()[pick],
            "MONTHS_BALANCE": -rng.integers(1, 96, nm),
            "AMT_BALANCE": rng.uniform(0, 1e5, nm).round(1),
            "AMT_CREDIT_LIMIT_ACTUAL": rng.choice([0.0, 45000.0, 90000.0], nm),
            "SK_DPD": rng.choice([0, 0, 5], nm),
        }
    ).to_csv(hc / "credit_card_balance.csv", index=False)
    return root
