-- One row per Home Credit application (train + Kaggle test): application fields, ratios and history.
-- Every history feature uses only records dated before the application (dbt tests on staging).
select
    a.*,
    a.credit_amount / nullif(a.income, 0)      as credit_to_income,
    a.annuity / nullif(a.income, 0)            as annuity_to_income,
    a.credit_amount / nullif(a.annuity, 0)     as credit_term_years,
    -a.days_birth / 365.25                      as age_years,
    coalesce(b.bureau_n_loans, 0)              as bureau_n_loans,
    coalesce(b.bureau_n_active, 0)             as bureau_n_active,
    b.bureau_total_debt,
    b.bureau_total_credit,
    b.bureau_total_debt / nullif(b.bureau_total_credit, 0) as bureau_debt_to_credit,
    b.bureau_total_overdue,
    b.bureau_max_days_overdue,
    b.bureau_max_amt_overdue,
    b.bureau_n_prolong,
    b.bureau_days_since_last_open,
    b.bureau_days_since_first_open,
    b.bureau_worst_dpd_bucket,
    b.bureau_worst_dpd_bucket_12m,
    coalesce(p.prev_n_applications, 0)         as prev_n_applications,
    coalesce(p.prev_n_refused, 0)              as prev_n_refused,
    coalesce(p.prev_n_approved, 0)             as prev_n_approved,
    p.prev_mean_credit_to_application,
    p.prev_days_since_last_decision,
    p.inst_n,
    p.inst_share_late,
    p.inst_mean_days_late,
    p.inst_share_underpaid,
    p.inst_share_unpaid,
    p.pos_max_dpd,
    p.pos_mean_dpd_def,
    p.cc_max_dpd,
    p.cc_mean_utilisation,
    p.cc_mean_utilisation_6m
from {{ ref('stg_hc__application') }} a
left join {{ ref('int_hc__bureau_by_applicant') }} b using (sk_id_curr)
left join {{ ref('int_hc__previous_by_applicant') }} p using (sk_id_curr)
