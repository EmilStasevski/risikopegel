-- Sanity bound on the training default rate (8.07% in the full data); catches broken casts/joins.
select avg(target) as rate
from {{ ref('mart_hc__applicant_features') }}
where is_train
having avg(target) not between 0.01 and 0.5
