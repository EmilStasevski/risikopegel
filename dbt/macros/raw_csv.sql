{# Read a raw CSV from var('raw_dir'). GMSC writes missing values as 'NA', Home Credit as empty fields. #}
{% macro raw_csv(relpath) -%}
read_csv('{{ var("raw_dir") }}/{{ relpath }}', header = true, sample_size = -1, nullstr = ['NA', ''])
{%- endmacro %}
