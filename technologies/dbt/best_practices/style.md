# dbt (Data Build Tool) Best Practices - Style

- [Back to Table of Contents](toc.md)

Pulled from: https://docs.getdbt.com/best-practices/how-we-style/0-how-we-style-our-dbt-projects

## Main Ideas:

### Clarity and Consistency

- Able to quickly read and understand code

---

## [Model and Field Naming](https://docs.getdbt.com/best-practices/how-we-style/1-how-we-style-our-dbt-models)

### Models

- **Pluralized**
- **Must have primary key named [object]\_id**: this way it is always easy to know what the id references
- **Use_underscores_in_names**
- Keys should be **`str` types**
- **Do NOT use abbreviations for aliases**: Emphasize _readability_ over brevity (i.e. don't use `o` for `orders`)
- **Avoid reserved words as column names**:

### All

- **Use `snake_case`**: for schema, table, and column names (and probably most things to be consistent)
- Booleans should be prefixed with `is\_` or `has\_` (i.e. `has_insurance`)
- Timestamps should be named `[event]\_at` AND be in **UTC** (i.e. `updated_at`)
- Dates should be named `[event]\_date` (i.e. `created_date`)
- Dates and times should be **past tense**
- Prices should be in **decimal currency** (i.e. `19.99`). **If a non-decimal currency is used**, indicate with a suffix (`price_in_cents`)
- Use **BUSINESS TERMINOLOGY** not the source terminology. If a source calls them a `user` but the business calls them a `consumer`, use `consumer`
- Indicate versions of models with a suffix of `_v1` or `_v2`
- In models, keep data types together when defining to make things more understandable:

```
with source as (
  select * from {{ source('ecom', 'raw_orders') }}
),
renamed as (
    select

        ----------  ids
        id as order_id,
        store_id as location_id,
        customer as customer_id,

        ---------- strings
        status as order_status,

        ---------- numerics
        (order_total / 100.0)::float as order_total,
        (tax_paid / 100.0)::float as tax_paid,

        ---------- booleans
        is_fulfilled,

        ---------- dates
        date(order_date) as ordered_date,

        ---------- timestamps
        ordered_at

    from source
)
select * from renamed
```

---

## [SQL](https://docs.getdbt.com/best-practices/how-we-style/2-how-we-style-our-sql)

### Basics

- Use **SQLFluff** to automatically abide by these rules (`pip install sqlfluff`)
- Use Jinja comments `{# #}` for comments not to include in compiled code
- Use trailing commas
- Four spaces for an indent
- Max 80 char length per line
- Field names, keywords, and function names should all be **lowercase**
- `as` keyword should be used **explicitly when aliasing**

### Fields, Aggregations, and Grouping

- State fields prior to aggregations and window functions
- Run aggregations as early as possible on the smallest data set possible before joining to improve performance
- `order by` and `group by` a number (`group by 1`) is **preferred over listing column names**
  - In `group by 1`, `1` refers to the first column in the select statement

### Joins

- Use `union all` over `union` unless you explicitly want to remove duplicates
- If using a `join`, **always prefix column names with the table name**
- Use specific joins over `join` (i.e. `inner join`, `outer join`)
- Avoid using aliases
- Move **left to right** to make join reasoning easy to understand. If using a `right join`, change up the `from` and `join` tables

### _Import_ CTEs (top of file _ref_)

- All `{{ ref ('...') }}` statements should be in the CTEs at the top of the file
- Named after the table referencing
- Only select columns you're actually using and use where if possible to filter out unnecessary data

```
with orders as (
    select
        order_id,
        customer_id,
        order_total,
        order_date

    from {{ ref('orders') }}

    where order_date >= '2020-01-01'
)
```

### _Functional_ CTEs

- These should try to do a single unit of work (if performance permits)
- Use verbose names to easily understand what it is doing
- Duplicated CTEs should be pulled into their own `intermediate` models and referenced from there (i.e. repeated selecting and trimming of a dataset)
- Last line of a model should always be `select *`

---

## [Jinja](https://docs.getdbt.com/best-practices/how-we-style/4-how-we-style-our-jinja)

- When using, add spaces inside your delimiters like `{{ this }}` and not `{{this}}`
- Use newlines to visually indicate logical blocks of Jinja
- Indent 4 spaces into a Jinja block to visually indicate code inside is wrapped in a block

---

## [YAML](https://docs.getdbt.com/best-practices/how-we-style/5-how-we-style-our-yaml)

- Indent 2 spaces
- List itens should be indented
- List item best practice is to provide the argument as a list: `'select': ['other_user']`
- Use newlines to separate dictionaries
- Max 80 char length per line
