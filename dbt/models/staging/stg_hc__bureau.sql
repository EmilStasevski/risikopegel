-- Credit-bureau records of other lenders' loans, one row per SK_ID_BUREAU.
-- Debt/overdue amounts are as of DAYS_CREDIT_UPDATE; rows updated after the application (17 in the full
-- data) would leak post-application information and are dropped.
select
    cast(SK_ID_CURR as integer)            as sk_id_curr,
    cast(SK_ID_BUREAU as integer)          as sk_id_bureau,
    CREDIT_ACTIVE                          as credit_active,
    CREDIT_TYPE                            as credit_type,
    cast(DAYS_CREDIT as integer)           as days_credit,          -- opened, relative to application (<0)
    cast(CREDIT_DAY_OVERDUE as integer)    as credit_day_overdue,
    cast(DAYS_CREDIT_ENDDATE as double)    as days_credit_enddate,  -- planned end: contract term, may be > 0
    cast(DAYS_ENDDATE_FACT as double)      as days_enddate_fact,
    cast(AMT_CREDIT_MAX_OVERDUE as double) as amt_max_overdue,
    cast(CNT_CREDIT_PROLONG as integer)    as n_prolong,
    cast(AMT_CREDIT_SUM as double)         as amt_credit_sum,
    cast(AMT_CREDIT_SUM_DEBT as double)    as amt_debt,
    cast(AMT_CREDIT_SUM_OVERDUE as double) as amt_overdue,
    cast(DAYS_CREDIT_UPDATE as integer)    as days_credit_update
from {{ raw_csv('homecredit/bureau.csv') }}
where DAYS_CREDIT_UPDATE <= 0 or DAYS_CREDIT_UPDATE is null
