{# Leakage guard (PLAN §0 rule 4): event-time columns are relative to the application date
   (negative = before). Fails on any row dated at or after the application. #}
{% test not_after_application(model, column_name, strict=true) %}
select *
from {{ model }}
where {{ column_name }} {{ '>=' if strict else '>' }} 0
{% endtest %}
