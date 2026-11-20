-- Previous Home Credit applications and their repayment behaviour per applicant.
with prev as (
    select
        sk_id_curr,
        count(*)                                                   as prev_n_applications,
        count(*) filter (where contract_status = 'Refused')        as prev_n_refused,
        count(*) filter (where contract_status = 'Approved')       as prev_n_approved,
        avg(amt_credit / nullif(amt_application, 0))               as prev_mean_credit_to_application,
        max(days_decision)                                         as prev_days_since_last_decision
    from {{ ref('stg_hc__previous_application') }}
    group by 1
),
inst as (
    select
        sk_id_curr,
        count(*)                                                                        as inst_n,
        avg(case when days_entry_payment > days_instalment then 1.0 else 0.0 end)       as inst_share_late,
        avg(greatest(days_entry_payment - days_instalment, 0))                          as inst_mean_days_late,
        avg(case when amt_payment < amt_instalment then 1.0 else 0.0 end)               as inst_share_underpaid,
        -- missing payment date with a due date before the application = unpaid installment
        avg(case when days_entry_payment is null then 1.0 else 0.0 end)                 as inst_share_unpaid
    from {{ ref('stg_hc__installments') }}
    -- a payment made after the application would be future information
    where days_entry_payment is null or days_entry_payment < 0
    group by 1
),
pos as (
    select sk_id_curr, max(dpd) as pos_max_dpd, avg(dpd_def) as pos_mean_dpd_def
    from {{ ref('stg_hc__pos_cash') }}
    group by 1
),
cc as (
    select
        sk_id_curr,
        max(dpd)                                                            as cc_max_dpd,
        avg(amt_balance / nullif(amt_limit, 0))                             as cc_mean_utilisation,
        avg(case when months_balance >= -6 then amt_balance / nullif(amt_limit, 0) end) as cc_mean_utilisation_6m
    from {{ ref('stg_hc__credit_card') }}
    group by 1
),
ids as (
    select sk_id_curr from prev
    union select sk_id_curr from inst
    union select sk_id_curr from pos
    union select sk_id_curr from cc
)
select *
from ids
left join prev using (sk_id_curr)
left join inst using (sk_id_curr)
left join pos using (sk_id_curr)
left join cc using (sk_id_curr)
