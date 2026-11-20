select
    cast(SK_ID_PREV as integer)    as sk_id_prev,
    cast(SK_ID_CURR as integer)    as sk_id_curr,
    cast(MONTHS_BALANCE as integer) as months_balance,
    cast(SK_DPD as integer)        as dpd,
    cast(SK_DPD_DEF as integer)    as dpd_def
from {{ raw_csv('homecredit/POS_CASH_balance.csv') }}
where MONTHS_BALANCE < 0
