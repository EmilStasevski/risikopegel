-- Home Credit applications, train + Kaggle test (target null). One row per SK_ID_CURR.
-- DAYS_EMPLOYED = 365243 is a placeholder (55,374 train rows, mostly pensioners) -> null + flag.
{% set cols %}
    SK_ID_CURR, NAME_CONTRACT_TYPE, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN,
    AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
    NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, REGION_POPULATION_RELATIVE, DAYS_BIRTH, DAYS_EMPLOYED,
    DAYS_REGISTRATION, DAYS_ID_PUBLISH, OWN_CAR_AGE, OCCUPATION_TYPE, CNT_FAM_MEMBERS,
    REGION_RATING_CLIENT, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_LAST_PHONE_CHANGE,
    AMT_REQ_CREDIT_BUREAU_YEAR
{% endset %}
with train as (
    select {{ cols }}, cast(TARGET as integer) as target, true as is_train
    from {{ raw_csv('homecredit/application_train.csv') }}
),
test as (
    select {{ cols }}, cast(null as integer) as target, false as is_train
    from {{ raw_csv('homecredit/application_test.csv') }}
),
u as (select * from train union all by name select * from test)
select
    cast(SK_ID_CURR as integer)               as sk_id_curr,
    is_train,
    target,
    NAME_CONTRACT_TYPE                        as contract_type,
    CODE_GENDER                               as gender,          -- protected-like attribute: fairness checks only
    FLAG_OWN_CAR = 'Y'                        as owns_car,
    FLAG_OWN_REALTY = 'Y'                     as owns_realty,
    cast(CNT_CHILDREN as integer)             as n_children,
    cast(AMT_INCOME_TOTAL as double)          as income,
    cast(AMT_CREDIT as double)                as credit_amount,
    cast(AMT_ANNUITY as double)               as annuity,
    cast(AMT_GOODS_PRICE as double)           as goods_price,
    NAME_INCOME_TYPE                          as income_type,
    NAME_EDUCATION_TYPE                       as education,
    NAME_FAMILY_STATUS                        as family_status,
    NAME_HOUSING_TYPE                         as housing_type,
    cast(REGION_POPULATION_RELATIVE as double) as region_population_rel,
    cast(DAYS_BIRTH as integer)               as days_birth,
    case when DAYS_EMPLOYED = 365243 then null else cast(DAYS_EMPLOYED as integer) end as days_employed,
    DAYS_EMPLOYED = 365243                    as employed_placeholder,
    cast(DAYS_REGISTRATION as double)         as days_registration,
    cast(DAYS_ID_PUBLISH as integer)          as days_id_publish,
    cast(OWN_CAR_AGE as double)               as own_car_age,
    OCCUPATION_TYPE                           as occupation,
    cast(CNT_FAM_MEMBERS as double)           as n_family_members,
    cast(REGION_RATING_CLIENT as integer)     as region_rating,
    cast(EXT_SOURCE_1 as double)              as ext_source_1,
    cast(EXT_SOURCE_2 as double)              as ext_source_2,
    cast(EXT_SOURCE_3 as double)              as ext_source_3,
    cast(DAYS_LAST_PHONE_CHANGE as double)    as days_last_phone_change,
    cast(AMT_REQ_CREDIT_BUREAU_YEAR as double) as n_bureau_requests_year
from u
