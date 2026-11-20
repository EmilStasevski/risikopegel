-- Give Me Some Credit modelling table. Special codes 96/98 nulled in the past-due counts for analysis
-- (the scorecard bins them as special codes from the raw columns instead).
select
    *,
    case when dpd_special_code then null else n_30_59_dpd end as n_30_59_dpd_clean,
    case when dpd_special_code then null else n_60_89_dpd end as n_60_89_dpd_clean,
    case when dpd_special_code then null else n_90_dpd end    as n_90_dpd_clean,
    monthly_income is null                                    as income_missing
from {{ ref('stg_gmsc__applicants') }}
