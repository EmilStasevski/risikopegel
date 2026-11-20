-- Bureau history per applicant (only records dated before the application, see staging).
with worst as (
    select sk_id_bureau, max(dpd_bucket) as worst_dpd_bucket,
           max(case when months_balance >= -12 then dpd_bucket else 0 end) as worst_dpd_bucket_12m
    from {{ ref('stg_hc__bureau_balance') }}
    group by 1
)
select
    b.sk_id_curr,
    count(*)                                             as bureau_n_loans,
    count(*) filter (where b.credit_active = 'Active')   as bureau_n_active,
    sum(b.amt_debt)                                      as bureau_total_debt,
    sum(b.amt_credit_sum)                                as bureau_total_credit,
    sum(b.amt_overdue)                                   as bureau_total_overdue,
    max(b.credit_day_overdue)                            as bureau_max_days_overdue,
    max(b.amt_max_overdue)                               as bureau_max_amt_overdue,
    sum(b.n_prolong)                                     as bureau_n_prolong,
    max(b.days_credit)                                   as bureau_days_since_last_open,  -- closest to 0
    min(b.days_credit)                                   as bureau_days_since_first_open,
    max(w.worst_dpd_bucket)                              as bureau_worst_dpd_bucket,
    max(w.worst_dpd_bucket_12m)                          as bureau_worst_dpd_bucket_12m
from {{ ref('stg_hc__bureau') }} b
left join worst w using (sk_id_bureau)
group by 1
