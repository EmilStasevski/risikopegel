-- The mart must keep every application (joins must not drop or duplicate rows).
select 'row count mismatch' as problem
where (select count(*) from {{ ref('mart_hc__applicant_features') }})
   <> (select count(*) from {{ ref('stg_hc__application') }})
