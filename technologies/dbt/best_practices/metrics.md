# dbt (Data Build Tool) Best Practices - Metrics

- [Back to Table of Contents](toc.md)

> Disclaimer: docs.getdbt.com best practices for defining metrics are now done through their _Semantic layer_ available in the dbt Cloud. If using dbt Core only, they provide no recommendations. The best practices below have been pulled from multiple other sources.

## Where should I define my metrics?

This is a decision that must be made by the business based on several considerations...

### Data Warehouse

- Metrics defined in a data warehouse can ensure **data integrity** as they are controlled and calculated in the warehouse with the raw data
- Metrics are **consistent** this way also. A `booked_appointment` means the same thing to everyone rather than having different definitions/calculations in different reports
- There are **performance and scalability advantages** here also as data can be pre-aggregated and stored
- Metrics defined here are also **reusable** across various tools, dashboards, reports

### BI Tool

- It is generally very easy to _drag and drop_ dimensions and values around in a BI tool creating a very **flexible and agile** way of making _ad-hoc_ analyses without having to write SQL queries
- Many BI tools allow users to dive in and dig deeper into metrics in real-time making it more **interactive**, though it is possible to still do this with metrics calculated at the data warehouse level
- If there are complex calculations, aggregations, or transformations that are easily handled by the BI tool (rather than `dbt`) it may make sense to define those in the BI tool

## Best Practices for Defining Metric Models in Data Warehouses

### Understanding Business Requirements and Data Usage

- **Types** of reports needed
- How **often the data** is accessed
- How **much data** is being accessed
- What are the desired **metrics/KPIs**?
- How does the business **DEFINE those metrics and KPIs**?

### Data Marts and Metrics

- Each mart allows you to segment your data based on various aspects for business units, departments, or user groups/**cohorts**.
- By pre-aggregating your metrics into a mart, you can optimize the performance of the reporting tools using your mart
- Marts can also provide a tier of access that limits visibility to lower-level data and acts as a security layer
- The data from marts can be customized in BI tools depending on the department's needs
- _Documentation is EXTREMELY important_: make sure that metric definition, calculation logic, source data, and any transformations are listed for the metric. That way, anyone using the data knows how it was calculated and where it came from (and developers in the future know how to ensure it's working correctly).
