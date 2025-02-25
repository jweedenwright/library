# dbt (Data Build Tool) Best Practices

> Pulled from: https://docs.getdbt.com/best-practices

**The need to establish a cohesive arc moving data from _source-conformed_ to _business-conformed_**

- **_Source-conformed_ data is shaped by external systems out of our control**
- **While _business-conformed_ data is shaped by the needs, concepts, and definitions we create**

**This process remains the essential purpose of the transformation layer**

---

## How dbt structures THEIR Projects

- It's crucial to establish consistent and comprehensible norms such that your team’s limited bandwidth for decision making can be spent on unique and difficult problems, not deciding where folders should go or how to name files.
- **FOLDER STRUCTURE IS CRITICAL** to find our way around the codebase and understanding the knowledge graph encoded in our project (alongside the DAG (Directed Acyclic Graph showing modeling, dependencies) and the data output into our warehouse) By using folders, and breaking items out, it makes it possible to rebuild all the affected components that build on that data if there is a change: `dbt build --select staging.oxygen+` (in this example, if `oxygen` changes, by adding the `+` to the build, it will build everything that uses `oxygen`)

---

## [Staging | Atomic Building Blocks](https://docs.getdbt.com/best-practices/how-we-structure/2-staging)

- Creating our atoms, our initial modular building blocks, from source data

```
models/staging
├── jaffle_shop
│   ├── _jaffle_shop__docs.md
│   ├── _jaffle_shop__models.yml
│   ├── _jaffle_shop__sources.yml
│   ├── base
│   │   ├── base_jaffle_shop__customers.sql
│   │   └── base_jaffle_shop__deleted_customers.sql
│   ├── stg_jaffle_shop__customers.sql
│   └── stg_jaffle_shop__orders.sql
└── stripe
    ├── _stripe__models.yml
    ├── _stripe__sources.yml
    └── stg_stripe__payments.sql
```

### Files and Folders

- **Sub directories should be based on source system** - We've found this to be the best grouping for most companies, as _source systems tend to share similar loading methods and properties between tables_, and this allows us to operate on those similar sets easily.
- **Filenames**: `[layer_prefix_stg]\_[source]\_\_[entity]s.sql` - They should include a prefix for the layer the model exists in, important grouping information, and specific information about the entity or transformation in the model.
  - **Use PLURALS** - match with prose. Unless their is only one _order_ in your table, it should be named _Orders_

### Models

- _Every staging model should follow this pattern: 2 CTES that..._
  - Pulls in a source table via the source macro
  - Apply our transformations
- _The **ONLY** place where the `source` macro is used_
- _Standard transformation types are handled here_
  - Renaming
  - Type casting
  - Basic computations (e.g. cents to dollars)
  - Categorizing (using conditional logic to group values into buckets or booleans, using a `successful` status as true)
- _Avoid Joins in Staging_- we are creating a modular component in staging (our atom) and joins will complicate and create additional computation/confusing relationships dowstream that are better handled elsewhere
- _Avoid Aggregations in Staging_- again, this is our atom/building block. By aggregating, you start losing access to some of the source data that is likely needed in other places
- **Materialized as VIEWS** - this is for 2 key reasons:
  1. A view will pull the latest data so all downstream models will get the latest and greatest
  2. Space is not wasted in the warehouse on these that are not intended for downstream consumers that need performance
- _1:1 Mapping of Source Table to Staging Tables_
- _Handle all major transformations as early upstream as possible_ - this reduces complexity and processing needed downstream

#### Exceptions

- As in the heirarchy above, if there is a case where it makes sense to combine tables to create a single atomic block (i.e. combining the tables `customers` and `deleted_customers` to create a single `customers` table), use a `base` directory to house those models.
- Unions can make sense when you have disparate data coming in with the exact same schemas (i.e. shopify data from different territories)

#### Helpers

- Once you've mastered this practice, there's a tool called [codegen](https://github.com/dbt-labs/dbt-codegen) that is available to automatically build models from source. This is recommended for on-going automation and consistency reasons.

#### Example Model:

```
with source as (
    select * from {{ source('<db>','<table>') }}
),

renamed as (
    select
        -- ids
        id as <table_id>,
        -- strings
        -- numerics
        -- booleans
        -- dates
        -- timestamps
    from source
)

select * from renamed
```

---

## [Intermediate | Purpose-built Transformation Steps](https://docs.getdbt.com/best-practices/how-we-structure/3-intermediate)

- stacking layers of logic with clear and specific purposes to prepare our staging models to join into the entities we want

```
models/intermediate
└── finance
    ├── _int_finance__models.yml
    └── int_payments_pivoted_to_orders.sql
```

### Files and Folders

- **Sub directories should be based on business groupings** - we split our models up into subdirectories by their area of business concern.
- **Filenames** `[layer_prefix_int]\_[entity]s\_[verb]s.sql` - Verbs such as `pivoted`, `aggregated_to_user`, `joined`, `fanned_out_by_quantity`, `funnel_created`. Source systems should be mostly removed at this stage, but if necessary, should retain the double underscores (`stg_shopify__orders_summed`)
- **Desire to bring sources together into a _single source of truth_ BEFORE sharing out to finance, marketing, etc** - you may not need sub-directories here if your business isn't overly complex (i.e. don't over-engineer it)

### Models

- _Every staging model should serve a clear **SINGLE** purpose_
- _Keep things DRY_ - don't repeat yourself and use `jinja` where possible to reduce complexity
- _Each file and transformation tells a story of how the data changes over time (via the DAG)_ dbt's Directed Acyclic Graph (DAG) showing modeling and dependencies should tell a story of how we handle the data

#### Example Model:

```
{%- set payment_methods = ['bank_transfer','credit_card','coupon','gift_card'] -%}
with payments as (
   select * from {{ ref('stg_stripe__payments') }}
),

pivot_and_aggregate_payments_to_order_grain as (
   select
      order_id,
      {% for payment_method in payment_methods -%}
         sum(
            case
               when payment_method = '{{ payment_method }}' and
                    status = 'success'
               then amount
               else 0
            end
         ) as {{ payment_method }}_amount,
      {%- endfor %}
      sum(case when status = 'success' then amount end) as total_amount
   from payments
   group by 1
)
select * from pivot_and_aggregate_payments_to_order_grain
```

---

## Marts

- bringing together our modular pieces into a wide, rich vision of the entities our organization cares about

## Utilities

- Helpful general purpose models that we generate from macros or based on seeds that provide tools to **help us do our modeling**, rather than data to model itself. The most common use case is a _date spine_ generated with the dbt utils package.

## Other folders in a dbt project: tests, seeds, and analyses
