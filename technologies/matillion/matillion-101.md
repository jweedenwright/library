# [Matillion](https://www.matillion.com/)

* Tabs vs highlighted files - TAB is your target, not what happens to be selected in the aside
* Box top right - who else is in the project
* Default = Main branch
    * Project Version - replica of main branch in github
* Project -> Manage Version -> Click on button in row of version you want to switch to
* Components -> UI components for building the pipeline
    * Click on a component and gives you all parameters for that component
* Transformation Job (`T_CDM_DIAGNOSIS`) - **GREEN**
    * Right-click and click on `Job Variable` to see the job variables
* Orchestration Jobs (`O_RUN_ALL_CDM`) - **BLUE**
    * Orchestration job of orchestration jobs that run transformation jobs
* Right-Click - `Validate Job`
    * Basically validates all is good (runs a _build_)
* Project -> Manage Environment Variables (for entire Matillion environment)
* Project -> Manage Project Group Passwords (Secret Manager)
* Job Variable - if blank, usually the job sets that value (or value could get passed)

