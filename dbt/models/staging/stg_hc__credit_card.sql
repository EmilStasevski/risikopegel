select
    cast(SK_ID_PREV as integer)              as sk_id_prev,
    cast(SK_ID_CURR as integer)              as sk_id_curr,
    cast(MONTHS_BALANCE as integer)          as months_balance,
    cast(AMT_BALANCE as double)              as amt_balance,
    cast(AMT_CREDIT_LIMIT_ACTUAL as double)  as amt_limit,
    cast(SK_DPD as integer)                  as dpd
from {{ raw_csv('homecredit/credit_card_balance.csv') }}
where MONTHS_BALANCE < 0
