# dbt (Data Build Tool) Best Practices - Structure

- [Back to Table of Contents](toc.md)

Pulled from: <https://docs.getdbt.com/best-practices>

## Some Main Ideas:

### Establish a cohesive arc moving data from _source-conformed_ to _business-conformed_

**This process remains the essential purpose of the transformation layer**

- **_Source-conformed_ data is shaped by external systems out of our control**
- **While _business-conformed_ data is shaped by the needs, concepts, and definitions we create**

### Narrow the Dag, Widen the Tables

Generally, having multiple **INPUTS** into a model is expected, but **NOT** outputs. This way we build up to `mart` models with a robust set of data that can quickly and easily be accessed and used to answer questions.

### Troubleshoot via Tables

While ideal in production, lots of ephemeral queries and views are not great when debugging in development. You may want to setup different `dbt_project.yml` files for `dev` vs `prod` so that some tables are materialized in one but not the other.

## How dbt structures THEIR Projects

- It's crucial to establish consistent and comprehensible norms such that your team's limited bandwidth for decision making can be spent on unique and difficult problems, not deciding where folders should go or how to name files.
- **FOLDER STRUCTURE IS CRITICAL** to find our way around the codebase and understanding the knowledge graph encoded in our project (alongside the DAG (Directed Acyclic Graph showing modeling, dependencies) and the data output into our warehouse) By using folders, and breaking items out, it makes it possible to rebuild all the affected components that build on that data if there is a change: `dbt build --select staging.oxygen+` (in this example, if `oxygen` changes, by adding the `+` to the build, it will build everything that uses `oxygen`)

--------------------------------------------------------------------------------

## [Staging Layer | Atomic Building Blocks](https://docs.getdbt.com/best-practices/how-we-structure/2-staging)

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

- **Sub directories should be based on source system**: We've found this to be the best grouping for most companies, as _source systems tend to share similar loading methods and properties between tables_, and this allows us to operate on those similar sets easily.
- **Filenames**: `[layer_prefix_stg]\_[source]\_\_[entity]s.sql` - They should include a prefix for the layer the model exists in, important grouping information, and specific information about the entity or transformation in the model.

  - **Use PLURALS**: match with prose. Unless their is only one _order_ in your table, it should be named _Orders_

### Models

- _Every staging model should **follow this pattern:** 2 CTES that..._

  - Pull in a source table via the source macro
  - Apply our transformations

- **ONLY place where the `source` macro is used**
- _**Standard transformations** are handled here_

  - Renaming
  - Type casting
  - Basic computations (e.g. cents to dollars)
  - Categorizing (using conditional logic to group values into buckets or booleans, using a `successful` status as true)

- **Staging Models should avoid...**

  - **Joins**: we are creating a modular component in staging (our atom) and joins will complicate and create additional computation/confusing relationships dowstream that are better handled elsewhere
  - **Aggregations**: again, this is our atom/building block. By aggregating, you start losing access to some of the source data that is likely needed in other places

- **Materialized as VIEWS**: this is for 2 key reasons:

  1. A view will pull the latest data so all downstream models will get the latest and greatest
  2. Space is not wasted in the warehouse on these that are not intended for downstream consumers that need performance

- **1:1 Mapping of Source Table to Staging Tables**
- _Handle all major transformations **as early upstream as possible**_: this reduces complexity and processing needed downstream

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

--------------------------------------------------------------------------------

## [Intermediate Layer | Purpose-built Transformation Steps](https://docs.getdbt.com/best-practices/how-we-structure/3-intermediate)

- stacking layers of logic with clear and specific purposes to prepare our staging models to join into the entities we want

```
models/intermediate
└── finance
    ├── _int_finance__models.yml
    └── int_payments_pivoted_to_orders.sql
```

### Files and Folders

- **Sub directories should be based on business groupings**: we split our models up into subdirectories by their area of business concern.
- **Filenames** `[layer_prefix_int]\_[entity]s\_[verb]s.sql`: Verbs such as `pivoted`, `aggregated_to_user`, `joined`, `fanned_out_by_quantity`, `funnel_created`. Source systems should be mostly removed at this stage, but if necessary, should retain the double underscores (`stg_shopify__orders_summed`)
- **Desire to bring sources together into a _single source of truth_ BEFORE sharing out to finance, marketing, etc**: you may not need sub-directories here if your business isn't overly complex (i.e. don't over-engineer it)

### Models

- **Materialized as ephemeral**: as these models are _generally_ not used by end users, their is no need to clog up the warehouse with unnecessary models (though this can impact debugging if there is an issue and you are unable to dig into these model results)
- _Every staging model should serve a clear **SINGLE PURPOSE**_
- _Keep things **DRY**_: don't repeat yourself and use `jinja` where possible to reduce complexity
- _Each file and transformation tells a story of how the data changes over time (via the DAG)_ dbt's Directed Acyclic Graph (DAG) showing modeling and dependencies should tell a story of how we handle the data

### Purposes

- **Structural Simplification**

  - Bringing 4-6 models/entities/concepts together in a single model in staging and joining with a similar table in a `mart` is much less complex than having 10 joins in a `mart`. It makes more sense to have 2 smaller, readable, testable components than to have a massive monolith of a model in a `mart`

- **Re-graining**

  - Often used to fan out or collapse to the correct grain (i.e. a row for each order, summarizing orders by employee)

- **Isolating Complex Operations**

  - This makes the models easier to refine, troubleshoot, and simplifies later models that reference this complex concept in a clearly readable way.

#### Exceptions

Intermediate models can be _materialized as views in a custom schema with special permissions_. This can help with insight in development and troubleshooting while still keeping them separated from models used by end users. Just keep your warehouse tidy!

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

--------------------------------------------------------------------------------

## [Mart (Entity/Concept) Layer | Business-defined Entities](https://docs.getdbt.com/best-practices/how-we-structure/4-marts)

- stacking layers of logic with clear and specific purposes to prepare our staging models to join into the entities we want
- Entities such as order, customer, territory, click event, or payment are represented here in a distinct mart with each row containing all instances of these entities.
- Storage is cheap and compute is expensive, so happily building the same data in multiple places is more efficient than repeatedly rejoining/computing the dataset

```
models/marts
├── finance
│   ├── _finance__models.yml
│   ├── orders.sql
│   └── payments.sql
└── marketing
    ├── _marketing__models.yml
    └── customers.sql
```

### Note on the dbt Semantic layer

`dbt` offers a _Semantic_ layer in the `dbt Cloud` solution. In the _Semantic Layer_(<https://docs.getdbt.com/docs/use-dbt-semantic-layer/dbt-sl>) ([FAQs](https://docs.getdbt.com/docs/use-dbt-semantic-layer/sl-faqs)) users are able to create dynamic SL to compute metrics as well as define custom metrics. In this case, you'd want your `mart` level to be as **normalized** as possible. The rest of this readme will assume it is **NOT** being used.

### Files and Folders

- **Sub directories should be based on business groupings**: we split our models up into subdirectories by their area of business concern. If you only have a few marts though, don't feel that you need to use sub directories.
- **Filenames** `[entity]s.sql`: If a `mart/entity` doesn't contain a time-based value, ensure that the time is reflected in the name such as `orders_per_day` or `appts_by_month`
- **Avoid creating the same concept differently for different teams**: This `anti-pattern` can get confusing and complex. If you need to have different models, it should be around a separate concept (such as reporting revenue to government vs to the board). _**It should NOT be departmental views of the same concept**_

### Models

- **Materialized as Tables or incremental models**: by having these as a table in our Data Warehouses, it reduces costly recompute of all the chains of models we've created to get here whenever a user refreshes their dashboard.

  - A good practice is to use a **view** until it takes too long to practically query, then a **table**...and then an **incremental model**

- **Wide and denormalized**: with cheap storage and compute expensive, all the data should be ready to go and available
- **Avoid too many Joins**: As mentioned in the Intermediate section above, reducing complexity is extremely important when it comes to readability and building a clear mental model of what is happening.
- **Build / Reuse separate marts _Thoughtfully_**: again, while the strategy is to get a narrow DAG, including a mart in another mart's generation may be necessary (such as using an `orders` mart as part of building the `customer` mart to get critical order data.)
- **Marts are based on entities, but that doesn't mean they don't contain _other_ entity data** - as in the last bullet, a `customer` entity may also contain `order` or `visit/session` data. Just keep away from building metrics in your marts (like `user_orders_per_day`)

#### Example Model:

```
-- orders.sql

with orders as  (
   select * from {{ ref('stg_jaffle_shop__orders' )}}
),

order_payments as (
    select * from {{ ref('int_payments_pivoted_to_orders') }}
),

orders_and_order_payments_joined as (
    select
        orders.order_id,
        orders.customer_id,
        orders.order_date,
        coalesce(order_payments.total_amount, 0) as amount,
        coalesce(order_payments.gift_card_amount, 0) as gift_card_amount
    from orders
    left join order_payments on orders.order_id = order_payments.order_id
)
select * from orders_and_payments_joined
```

--------------------------------------------------------------------------------

## Utilities

- Helpful general purpose models that we generate from macros or based on seeds that provide tools to **help us do our modeling**, rather than data to model itself. The most common use case is a _date spine_ generated with the dbt utils package.

--------------------------------------------------------------------------------

## YAML Structure

- Make configs as easy to find as possible
- Other than _top-level_ files such as `dbt_project.yml` and `packages.yml`, you can name, locate, and organize other `.yml` files however you want.

### Best Practices

- **One config / folder**: `_<dir_name>__models.yml` should configure all models in the directory

  - For `staging` you'll need one for sources also (per directory): `_<dir_name>__sources.yml`
  - The leading `_` ensures it's always at the top and easy to separate from models
  - This is considered the most balanced approach as it avoids a _monolythic single config for the entire project_ that is hard to manage and find information in as well as doesn't slow down development and add overly cumbersome busy work in the _single config per model_ approach.

- **For docs, follow the same pattern**: `_<dir_name>__docs.md` will ensure your documentation is also easy to find and consistent.
- **Cascade configs**: use `dbt_project.yml` to set default configurations at the directory level (i.e. materialization of `mart` models as tables) and then special conditions can be broken down at the model level. By doing this, we reduce on redundancy and needing to define multiple things in lower-level configs.

--------------------------------------------------------------------------------

## Other folders in a dbt project: tests, seeds, and analyses

### Seeds

- Most commonly used for lookup tables helpful for modeling but not in any source system (i.e. mapping zip codes to states)
- These should **NOT** be used for loading source data. You should be connecting to the source and loading the raw data into your warehouse whenever possible.

### Analysis

- Used for storing auditing queries that aren't built into your models. `dbtdocs` finds the most commonly used to store queries that leverage the audit helper package. This can be used for finding discrepancies in output when migrating logic from another system into `dbt`.

### Tests

- Most useful for testing multiple specific tables simultaneously (i.e. tables interacting with each other)
- Most of the time pre-built tests are all you need for single tables ([`dbt-expectations`](https://github.com/calogica/dbt-expectations?tab=readme-ov-file) is a good example)

## Macros

- Useful for creating reusable transformations that are done over and over
