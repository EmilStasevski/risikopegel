-- Give Me Some Credit, labelled file. One row per applicant. Codes 96/98 in past-due counts are
-- special codes (not counts): kept raw, flagged, and nulled in the *_clean columns.
with src as (
    select * from {{ raw_csv('givemesomecredit/cs-training.csv') }}
)
select
    cast(column00 as integer)                             as applicant_id,  -- unnamed first CSV column (DuckDB 2.0 naming)
    cast("SeriousDlqin2yrs" as integer)                    as default_2y,
    "RevolvingUtilizationOfUnsecuredLines"                 as revolving_util,
    cast(age as integer)                                   as age,
    cast("NumberOfTime30-59DaysPastDueNotWorse" as integer) as n_30_59_dpd,
    "DebtRatio"                                            as debt_ratio,
    cast("MonthlyIncome" as double)                        as monthly_income,
    cast("NumberOfOpenCreditLinesAndLoans" as integer)     as n_open_lines,
    cast("NumberOfTimes90DaysLate" as integer)             as n_90_dpd,
    cast("NumberRealEstateLoansOrLines" as integer)        as n_real_estate,
    cast("NumberOfTime60-89DaysPastDueNotWorse" as integer) as n_60_89_dpd,
    cast("NumberOfDependents" as integer)                  as n_dependents,
    ("NumberOfTimes90DaysLate" in (96, 98))                as dpd_special_code
from src
