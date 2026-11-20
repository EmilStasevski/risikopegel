-- Previous Home Credit applications. 365243 in DAYS_* due dates is a placeholder -> null.
select
    cast(SK_ID_PREV as integer)        as sk_id_prev,
    cast(SK_ID_CURR as integer)        as sk_id_curr,
    NAME_CONTRACT_TYPE                 as contract_type,
    NAME_CONTRACT_STATUS               as contract_status,
    cast(AMT_APPLICATION as double)    as amt_application,
    cast(AMT_CREDIT as double)         as amt_credit,
    cast(DAYS_DECISION as integer)     as days_decision
from {{ raw_csv('homecredit/previous_application.csv') }}
