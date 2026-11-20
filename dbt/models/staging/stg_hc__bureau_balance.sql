-- Monthly status of bureau loans. MONTHS_BALANCE = 0 is the application month and may extend past the
-- application date, so it is excluded (conservative; 610,965 rows in the full data).
select
    cast(SK_ID_BUREAU as integer)  as sk_id_bureau,
    cast(MONTHS_BALANCE as integer) as months_balance,
    STATUS                          as status,
    case when STATUS in ('1', '2', '3', '4', '5') then cast(STATUS as integer) else 0 end as dpd_bucket
from {{ raw_csv('homecredit/bureau_balance.csv') }}
where MONTHS_BALANCE < 0
