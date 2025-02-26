# dbt (Data Build Tool) Best Practices - Style

- [Back to Table of Contents](toc.md)

Pulled from: https://docs.getdbt.com/best-practices/how-we-style/0-how-we-style-our-dbt-projects

## Main Ideas:

### Clarity and Consistency

- Able to quickly read and understand code

---

## [Model and Field Naming](https://docs.getdbt.com/best-practices/how-we-structure/2-staging)

### Models

- **Pluralized**
- **Must have primary key named <object>\_id**: this way it is always easy to know what the id references
- **Use_underscores_in_names**
- Keys should be **`str` types**
- **Do NOT use abbreviations for aliases**: Emphasize _readability_ over brevity (i.e. don't use `o` for `orders`)
- **Avoid reserved words as column names**

### All

- **Use `snake_case`**: for schema, table, and column names (and probably most things to be consistent)
- Booleans should be prefixed with **is\_** or **has\_** (i.e. `has_insurance`)
- Timestamps should be named **<event>\_at** AND be in **UTC** (i.e. `updated_at`)
- Dates should be named **<event>\_date** (i.e. `created_date`)
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
