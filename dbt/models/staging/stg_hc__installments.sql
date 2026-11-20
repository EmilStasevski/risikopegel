-- Installment payments of previous Home Credit loans. Payment date missing = not (yet) paid.
select
    cast(SK_ID_PREV as integer)          as sk_id_prev,
    cast(SK_ID_CURR as integer)          as sk_id_curr,
    cast(DAYS_INSTALMENT as double)      as days_instalment,
    cast(DAYS_ENTRY_PAYMENT as double)   as days_entry_payment,
    cast(AMT_INSTALMENT as double)       as amt_instalment,
    cast(AMT_PAYMENT as double)          as amt_payment
from {{ raw_csv('homecredit/installments_payments.csv') }}
where DAYS_INSTALMENT < 0
