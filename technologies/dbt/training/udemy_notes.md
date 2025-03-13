# dbt Bootcamp

> Data Build Tool

## Section 2: Theory - The Data Maturity Model

### Maslow's Pyramid of Data

> Skipping a step will most likely cause us to fail in our goal and we will need to start back at the step we skipped.

1. Data Collection - capturing relevant data
2. Data Wrangling - cleaning / transforming
3. Data Integration - data lake / data warehouse

> The following aren't in this case

4. BI and Analytics
5. AI

### The Data Maturity Model

- Data Collection - api, database, images, videos
  - 3 Vs: Variety, Velocity, and Volume
    - Critical when designing the infrastructure (understanding what types, how fast, and how much)
- Data Wrangling
  - Data moved into a _staging_ area
  - Data quality
  - Duplicate data, inconsistent values (zip codes, addresses, gender), missing data
  - **Transforming from operational source to data warehouse format**
- Data Integration
  - Writing our transformed data from staging to target database
  - Loading in 2 ways:
    - Refresh - rewrite/replace existing data completely
    - Update - only changes applied to the source will be applied to the data warehouse

### ETL and ELT

- ETL
  - First used process as database was expensive in the past, so transforms would occur _OUTSIDE_ the database in a staging area
  - Harder to transform / less reliable in filestore / staging areas before putting them in
- ELT - The new hotness
  - Databases / data warehouses are extremely performant and cheap to use, so putting data into
  - Easier to transform after loaded in DB than in a staging area

## Section 3: Theory - Data Warehouses, Data Lakes, and Lakehouses

### Data Warehouses (DW, DWH)

- Around since 1960s
- Dimensions, facts, handle denormalization
- Good with complex datatypes, columnlar data
- Can't handle unstructured data like images or videos
- SQL
- Used for reporting / dashboarding purposes

### External Tables and Cloud Data Warehouses

- Cloud options: Amazon Redshift, BigQuery, Snowflake
  - Easy to add/remove nodes, all maintenance/operations are taken care of in your fees for using
- On-Premise options: Oracle, IBM, teradata.
  - Harder to scale, expensive to setup, maintain, licenses, operational costs
- Concept of _external tables_
- Decouple compute from storage components

### Data Lakes

- Deal with unstructured or semistructured data
- Data Lake - repository of all kinds of data from clean, structure, unstructured
- Scalable file system - HDFS or Hadoop, Amazon S3, Azure Datalake Storage
- Store files so they have _no compute_
- Can scale _compute_ instances independently of _storage_
- Databricks / Snowflake do this _BY DEFAULT_

### Data Lakehouse

- Emerged due to limitations of data lakes
- Combines best features of data lakes and data warehouses
- Cost-efficient storage from provider
- Ensure schema of data in a lake house meta store
- Can control who has access to your data

## Section 4: Theory - The Modern Data Stack

### Modern Data Stack

- Evolution
- High storage prices led to this flow:
  - _OLD: ETL: Sources -> (EXTRACT) -> Staging Area (TRANSFORM) -> Data Warehouse (LOAD) - Analytics_
- _Data Warehouse Evolution_
  - Legacy - Symmetric Multi Processing Database (SMP)
    - Many processors share resources - memory, io, devices, data
    - Shared bus, parallel processing in CPU cores while resources are _SHARED_
    - Limitation to how much you can add to the same machine
    - Pros - faster processing times, all one same machine
    - Cons - only vertical scaling, need to keep backups
  - _MPP - Massively Parallel Processing / Multi PERLOFF_
    - Multiple nodes with own OS, memory
    - Main/Master node communicates with compute nodes using ODBC
    - _Storage can be shared or not_
      - _Share nothing MPP is most common MPP design_
      - Data from a single table might span many nodes, and may only contain specific rows for processing
    - _Example architectures:_ Azure Synapse SQL Pool, Amazon Redshift, Google BigQuery, Snowflake Computing Storage
  - _Decoupling of storage and compute_
    - Business / Analytics requests
      - Main node can take analytics and business requests and route them to analytics nodes for processing
      - Computing only required for these request
      - _Can shut down compute nodes when you don't need them_
      - Can add more compute if more requests come in
      - Scalable
  - _Column-oriented databases_
    - _Row-oriented databases_
      - Conventional technique, store data fast, good for writing/reading rules (PostgreSQL, MySQL)
      - Good for OLTP (online transactional processing)
      - Not good for analytical workloads
    - _Column-oriented databases_
      - Arrange data by fields, storing all data associated with a field in memory
      - Improve query performance by being optimized for column reading and computation
      - Good for OLAP (online analytical processing)
- Because of items above and cheaper storage, cheaper transistors, faster internet
  - _NEW: ELT: Sources -> (EXTRACT) -> (LOAD) -> Data Warehouse (TRANSFORM) -> Anaytics_
- Architecture
  - Data source: Google Analytics, Linked In, Shopify, Stripe
  - Extract / Load: Airbyte, _Fivetran_, Segment, _Stitch_
  - Transform: _dbt_, matillion, data iku (also data science)
    - Sits on top of Data Warehouse: _Snowflake_
  - BI Tools: had their own storage, acted as a Date Warehouse, large files, resource intensive
    - Modern use cloud connections now
- _dbt_ was created to do the _t (TRANSFORM)_ in the data warehouse

## Section 5: Theory - Slowly Changing Dimensions (SCD)

### The Basics

> Changes rarely and unpredictably, requiring a specific approach to handle referential integrity

### Type 0 - Retain Original (No Change)

- Applied to the _SOURCE_ table, but _NOT_ the _DATA WAREHOUSE_ Table as the original data is no longer worthwile to maintain anymore
  - _Example: Fax Numbers_
    - Once collected and important, now mostly irrelevant
    - No need to update in the _DATA WAREHOUSE_ (used for analytics/reporting) but still helpful on website (i.e. the source)

### Type 1 - Overwrite

- Dimension change, only the _NEW_ value is important
- Want to make sure the new value is moved to the _DATA WAREHOUSE_
- Apply to _BOTH_ _SOURCE_ and _DATA WAREHOUSE_
  - _Example: Added Air Conditioning_
    - Irrelevant that the unit at one point didn't have AC. What is important is that it now DOES

### Type 2 - Add New Row (All Historical Data)

- Both current AND history important to business
- Could be necessary to maintain for future validation / legal reasons
- New row in the DWH, so we get a whole VIEW of what happened
- _Example: Change in Rent_
  - Current price in the _SOURCE_
  - Detail analysis on prices, so the _DATA WAREHOUSE_ will keep ALL changes to the rent
- _BENEFIT_: all history is retained
- _CON_: processing speed can be high and costs can increase

### Type 3 - Add New Attribute (Some Historical Data)

- Keep _some_ historical data (like _previous_ vs _current_)
- Keep records lower, so processing time is lower
- _Example: Changes in Type_
  - _Private_ room to a _Entire_ room (but don't care about no longer valid types)
  - _Entire_ room to a _Private_ room
  - Only care about the _MOST RECENT_ type change in our _DATA WAREHOUSE_

## Section 6: Intro to the Practical Sessions: dbt and Airbnb Use-Case

### dbt Overview

- dbt = T (TRANSFORM) in ELT
- Examples:
  - A SQL statement is like a candle in a dark room, it gives you a glimpse at the room, but it goes out and you're in the dark again
  - _dbt_: A spotlight is portable, small and can show ALL parts of the room (and everyone can see the room)

> dbt will deploy deploy your analytics code following software engineering best practices such as modularity, portability, CICD, Testing, and Documentation

- dbt will allow you to build production-grade data pipelines
- Write code, dbt compiles to SQL, executes your data transformations again Snowflake or others
- Transformations are version-controlled, easily tested, dag graphs of all models are created
- Allows you to have different envs (such as dev, QA, prod) and quick switching
- Crafts dependency orders and streamline your queries

### Use-case and Input Data Model Overview

- Analytics engineer with AirBNB
- _Pipeline_ - Loading, cleansing, exposing data
- Writing tes, automations, and documentation
- Data source: inside AirBNB: Berlin
- `dbt` - `snowflake` - `preset` (BI tool to connect to snowflake)

#### Requirements

- Modeling changes are easy to follow and revert
- Explicit dependencies between models (order of operations for pipeline)
- Explore dependencies between models
- Data quality _tests_
- Error reporting
- Incremental load of fact tables
- Track history of dimension tables
- Easy-to-access documentation

Snowflake registration
dataset import
dbt installation
dbt setup, snowflake connection

## Section 7: Practice - Setup

- `dbt init <project-folder-to-create>`
- `dbt debug`

### Important Links

- One-click Snowflake/dbt setup: https://dbt-data-importer.streamlit.app/
- Bootcamp GitHub: https://github.com/nordquant/complete-dbt-bootcamp-zero-to-hero
- dbt Core: https://github.com/dbt-labs/dbt-core
- dbt Install: https://docs.getdbt.com/docs/core/connect-data-platform/snowflake-setup
- dbt configuration to connect to Snowflake: cz99677.us-east-2.aws, 3wl8jgtk.r.us-east-1, gxtakps-gu29841.snowflakecomputing.com
- Snowflake
  - Username: JWW
  - Dedicated Login URL: https://gxtakps-gu29841.snowflakecomputing.com

###

- Install: `pip install dbt-core dbt-snowflake`

### To Build

- airbnb.hosts -> src_hosts -> dim_hosts_cleansed -> dim_listings_w_hosts -> dashboard
- airbnb.listings -> src_listings -> dim_listsings_cleansed -^
- airbnb.reviews -> src_reviews -> fct_reviews -> mart_fullmoon_reviews -> dashboard
- seed_full_moon_dates -^
- Tests: dim_listings_minimum_nights, consistent_created_at, full_moon_no_sleep

## Section 8: Models

> Fundamental concepts behind dbt

### Overview

- Basic building blocks of your business logic
- SQL definitions that materialize as tables or views
- Stored as SQL files in the models file
- Models can reference other models
- Different scripts and macros

### Theory: CTE - Common Table Expressions

- Readable and maintable code
- Temporary result set
- Similar to a view, definition isn't stored in metadata
- Very readable, easy to maintain
- Example:
  - col_names is optional, just for aliases

```
--- STEP 1
WITH <name of result set> ([col_names]) AS (
  -- STEP 2
  <cte_query>
)
-- STEP 3
<reference_the_CTE>
```

### Creating our First Model: Airbnb Listings

- Staging layer:
  - src_listings
  - src_hosts
  - src_reviews
- By default, all of our models will be views

## Section 9: Materializations

> How models can be connected

### Overview

| Materialization  | When to Use                                            | When NOT to Use                                     |
| ---------------- | ------------------------------------------------------ | --------------------------------------------------- |
| view             | Lightweight representation. Don't reuse data too often | read from the same model several times              |
| table            | read from this model repeatedly                        | single-use models. Model is populated incrementally |
| incremental      | Fact tables. Appends to tables                         | You want to update HISTORICAL records               |
| ephemeral (CTEs) | You merely want an alias to your date.                 | You read from the same model several times          |

### Model Dependencies and dbt's ref tag

- `{{ref("<model>")}}` will reference an existing model

```
WITH src_hosts AS (
    SELECT * FROM {{ref("src_hosts")}}
)
```

### Table type materialization and project-level materialization config

- Default materialization is `view` but in the `dim` folder, it is `table`

```
models:
  dbtlearn:
    +materialized: view
    dim:
      +materialized: table
```

### Incremental Materialization

### Ephemeral Materialization

## Section 10: Seeds and Sources

### Overview

### Seeds

- local files that you upload to the data warehouse from dbt
- live in the `seeds` folder as `.csv` files

### Sources

- abstraction layer on top of your input tables
- once setup, able to check for freshness
- defined in `models/sources.yml`
- Once defined, can reference as `{{ source('airbnb', 'reviews') }}`

### Source Freshness

- can be checked automatically
- `dbt source freshness` - will check source freshness

## Section 11: Snapshots

### Overview

- Type-2 Slowly Changing Dimensions
  - Update an email address, want to keep history of the change
  - Takes original table, and adds new columns: `dbt_valid_from` and `dbt_valid_to`
  - Then adds a new row with the updated address
- 2 strategies for this
  - _Timestamp_: have a unique key and an updated_at field
  - _Check_: one or more columns can be checked and `dbt` will monitor

### Creating a Snapshot

- `dbt snapshot` will process snapshots
- Can snapshot any data
- Snapshotting RAW table so we can spot any changes coming in
  - _Note_: `invalidate_hard_deletes` means when an item is deleted, the snapshot is updated so that the `DBT_VALID_FROM` `null` is removed and set to the valid date. If `False`, the records are not updated

```
{% snapshot <snapshot_table_name> %}
{{
  config(
    target_schema='DEV',
    unique_key='id',
    strategy='timestamp',
    updated_at='updated_at'
    invalidate_hard_deletes=True
  )
}}
SELECT *
{% endsnapshot %}
```

## Section 12: Tests

### Overview

- 2 types of tests
  - _singular_: SQL queries stored in `tests/` folder expected to return an empty result set
  - _generic_: built in tests:
    - _unique_: ensure uniqueness of a column
    - _not_null_: ensure no null values
    - _accepted_values_: can pass a list of values and should be the only values in that column
    - _relationships_: make sure that a columns references to another table are all valid
    - Can write your own using `macros`

### Generic Tests

- `dim_listings_cleansed` - ensure id is unique and not null, listing names should be not null, room_type (accepted_values), host_id (reference to existing host id)
- `models/schema.yml` - setup extra configuration, tests, and documentation for your models

```
version:2

models:
  - name: dim_listings_cleansed
    columns:
    - name: listing_id
      tests:
        - unique
        - not_null
```

### Singular Tests

- Must return no results
- Live in the `tests` folder

## Section 13: Macros, Custom Tests, and Packages

### Overview

- jinja templates created in `macros` folder
- `dbt` already has a lot of built in
- `macro` named `test` for cureating custom tests

### Creating our First Macro

- `macro` keyword defines a macro follwed by the name of the macro and model name as param

```
{% macro no_nulls_in_columns(model) %}
    SELECT * FROM {{ model }} WHERE
    {% for col in adapter.get_columns_in_relation(model) -%}
        {{ col.column }} IS NULL OR
    {% endfor %}
    FALSE # terminates the OR statement above
{% endmacro %}
```

- `adapter.get_columns_in_relation(model)` - built in `dbt` function to get columns in the model
- Using a `-` at the end of a template block will cutout whitespace

### Writing Custom Generic Tests

- Codify a custom generated test in the macros folder - `postive_value`

```
{% test positive_value(model, column_name) %}
SELECT *
FROM {{ model }}
WHERE {{ column_name}} < 1
{% endtest %}
```

### Installing Third-Party Packages

- https://hub.getdbt.com/
- Helpful for generating unique ids: https://github.com/dbt-labs/dbt-utils/tree/1.3.0/#generate_surrogate_key-source
- Copy any libraries you want and add them to the project root's `packages.yml` file
- Install with `dbt deps`

## Section 14: Documentation

### Overview / Writing and Exploring Basic Documentation

- Defined in either `.yml` files or in `.md` files
- `dbt docs generate` - will build all documentation from yaml and md
- `dbt docs serve` - will create a local, lightweight docs server to view the documentation on

### Markdown-based Docs, Custom Overview Page and Assets

- Create an `.md` file under `models/` called `docs.md`
- Use `{% docs <title of doc> %} ... {% enddocs %}` to define a documentation block using markdown
- Use `{{ doc(<title of doc in MD>) }}` in the `schema.yml` to pull in `MD` description
- To override the Overview page, you'll need to create `overview.md` in the `models/` directory
- In the file, you add `{% docs __overview__ %}...{% enddocs %}` to override the overview
- LOCAL images, put in an `assets/` folder same directory as `models/`

### The Linage Graph (Data Flow DAG)

- Small icon floating in the bottom left allows you to view the flow diagram
- Can update with dropdowns in the bottom and click `Update Graph` to see the updates
- Can right click on any element in the flow to focus on or view documentation on it
- In the `-- select` drop down you can focus on a specific model, and then add `+` to show upstream dependencies and downstream models that use this model. For example: `+src_hosts+` would show both upstream and downstream. `+src_hosts` would only show upstream dependencies.
- In the `-- exclude` drop down you can remove specific items from the diagram as well as their parent and children. Example, to hide hosts and it's children: `src_hosts+`

## Section 15: Analyses, Hooks and Exposures

### Analyses

- SQL files created in the `analyses/` folder
- Execute adhoc query using `dbt` by running `dbt compile` and then getting the full query from the `target` directory

### Hooks

- SQL statements executed at predefined times
- _Hook Types_:
  - `on_run_start` - before a dbt run (BEFORE ALL MODELS)
  - `on_run_end` - after a dbt run (AFTER ALL MODELS)
  - `pre-hook` - before the execution of EVERY associated Model (FOR EACH MODEL)
  - `post-hook` - after the execution of EVERY associated Model (FOR EACH MODEL)
- In the following example, we use `post-hook` to give `SELECT` access to all models built on `dbt run` to the role of `REPORTER`

```
models:
  dbtlearn:
    +materialized: view
    +post-hook:
      - "GRANT SELECT ON {{ this }} TO ROLE REPORTER"
```

### Setting up a BI Dashboard in Snowflake and Preset

-

### Exposures

- added in `/models/` directory
- Can have any name (like `dashboards.yml`)
- Needs to use the `exposures` key term
- Used in documentation to describe all places where this data is EXPOSED externally

## Section 18: Debugging with Logging

- Log to logs/dbt.log
  - Just need to use jinja tag: `{{ log(<message>)}}`
- Log to screen
  - Just need to use jinja tag: `{{ log(<message>, info=True)}}`
- Disabling Macros
  - When a macro gets executed, the jinja part is ran first THEN the SQL, so you can't just comment out with SQL
  - Need to use a jinja comment: `{# log(<message>, info=True)#}`

## Section 19: Using Variables

- Two types:

  - jinja variables:

  ```
  {% set your_name_jinja = "Jeremiah" %}
  {{ log("Hello " ~ your_name_jinja, info=True )}}
  ```

  - dbt variables:

  ```
  {{ log("Hello dbt user " ~ var("user_name") ~ "!", info=True)}}
  ...
  dbt run-operation learn_variables --vars '{user_name: Jeremiah}'
  ```

- Check variable existence:

  ```
  {% if var("variable_name", False) %}
  ```

- Setting Defaults

  ```
  {{ log("Hello dbt user " ~ var("user_name", "NO USERNAME IS SET!!") ~ "!", info=True)}}
  ```

- Set default in `dbt_project.yml`
  ```
  vars:
    user_name: default_user_name_for_this_project
  ```
- Use Date Ranges to Make Incremental Models production-ready

  ```

  ```

## Section 20: Orchestrating dbt with Dagster

### Orchestration

- Nightly builds of tables, views, snapshots?

# Helpful Commands

- Libraries that we can use: https://hub.getdbt.com/
- Log to log file: `{{ log(<message>)}}`
- Log to screen: `{{ log(<message>, info=True)}}`
- Jinja comment: `{# log(<message>, info=True)#}`
- `dbt init <project-folder-to-create>`
  - Username: JWW
  - Dedicated Login URL: https://gxtakps-gu29841.snowflakecomputing.com
- `dbt debug`
- `dbt deps` - installs dependencies in `packages.yml`
- `dbt run --help` - Parameter listing!
- `dbt source freshness` - will check source freshness
- `dbt snapshot` - will run snapshots in `snapshots/` folder and create the `snapshot` table based on the query
  - Running this again will run an update on existing snapshot tables
- `dbt compile` - will check to ensure all code compiles without pushing to data warehouse
- `dbt test` - will run all tests (in `schema.yml` as well as the `tests` folder)
  - `dbt test --select <model>` - will run all tests that use this model
  - `dbt test --select source:airbnb.listings` - run a tests WITHIN the sources.yml
- `dbt docs generate` - will build all documentation from yaml and md
- `dbt docs serve` - will create a local, lightweight docs server to view the documentation on
- Folder structure for models:
  - `mart` folder is generally the layer that is accessible to BI tools
  - `dim` folder holds dimensional data (i.e. qualitative)
  - `fct` folder holds fact data (i.e. quantitative)
  - `src` are staged data files/ephemeral
- `dbt run --full-refresh` - will refresh ALL tables _INCLUDING incremental_
- `dbt run --select +<model>+` - will build only the model and either it's parents and/or children based on provided `+`
- `{{}}` - jinja SQL commands
- `{{ref("<model>")}}` will reference an existing model

```
WITH src_hosts AS (
    SELECT * FROM {{ref("src_hosts")}}
)
```

- `NVL` is a function that checks for null and gives a default value if it is

```
NVL(
    host_name,
    'Anonymous'
  ) AS host_name,
```

- `{{config}}` is used to setup the type of materialization and what to do on a schema change:

```
{{
  config(
    materialized = 'incremental',
    on_schema_change = 'fail'
    )
}}
```

- `{% if %}` is a jinja SQL conditional statement

```
{% if is_incremental() %}
  AND review_date > (select max(review_date) from {{this}})
{% endif %}
```
