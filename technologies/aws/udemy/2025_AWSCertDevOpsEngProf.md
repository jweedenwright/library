# AWS Certified DevOps Engineer Professional 2025
> Stephane Maarek
- Prerequisites: AWS Certified Developer course and certification
- AWS DevOps exam is HARD and tests you on real-world experience (probably should have a minimum of 2 years)

Referenced Courses and Lectures
- CCP: AWS Certified Cloud Practitioner
- SAA: AWS Certified Solutions Architect Associate
- DVA: AWS Certified Developer Associate
- SOA: AWS Certified SysOps Administrator Associate
- SAP: AWS Certified Solutions Architect Professional

**GOAL: End of 2025**

## CICD - Continuous Integration (Pushing Often)
- Learned how to create AWS resources 
- Interact with AWS programmatically (AWS CLI)
- Deploy code to AWS using Elastic Beanstalk
- All of these are MANUAL
- **Best Practice:** have code in a repository and then deployed onto AWS automatically, with the correct environment, tested, and manually approved where necessary
- AWS Services
    - CodeCommit
    - CodePipeline
    - CodeBuild
    - CodeDeploy
    - CodeStar
    - CodeArtifact
    - CodeGuru
- **CI**
    - Push code to github
    - Testing / build server checks as soon as pushed
    - The developer gets feedback
    - Find bugs early
    - Deliver faster (tested)
- **CD**
    - Ensures reliably released software, whenever needed
    - Build Server passes code along to deployment server
    - Pushes out to application servers - zone 1 for testing
    - Pushes out to app servers after manual approval?
    - CodeDeploy

- Technologies
| Title                 | Technology |
| ---                   | --- |
| Entire Flow           | Orchestrate with AWS CodePipeline |
| Code                  | Github |
| Build/Test            | AWS CodeBuild |
| Deploy + Provision    | AWS Elastic Beanstalk |
| Deploy                | AWS CodeDeploy |
| Provision             | CodeDeploy would set up EC2 / ECS / Lambda |

## LEGACY - Code Commit
- Can send events, like commit/merge, to EventBridge and invoke ECS task based on events
- Can setup IAM policies to prevent junior devs from committing to prod


## Elastic Beanstalk
- Used to create an APPLICATION which then has ENVIRONMENTS that are for that APPLICATION
- ENVIRONMENTS can be a WEB SERVER for a website/web app, or web API, or a worker app that does long-running workloads on demand/schedule
- Environment platforms can be .NET, Docker, PHP, Python, Node.js, Ruby, Java, and a few others

## CodePipeline - Overview (Alternative to CIRCLECI)
- Visual workflow tool to orchestrate CICD
- Source (github) -> Build (Codebuild) -> Test (CodeBuild) -> Deploy (CodeDeploy, beanstalk) -> Invoke (Lambda)
- Consists of STAGES, each can have sequential or parallel actions
- Stage can have MANUAL approval
- CodeBuild can use an extracted artifact from GitHub in S3 to build and stores built items in S3
- CodeDeploy - can use built artifacts in S3 to deploy them
- Debugging
   - CloudWatch (Amazon EventBridge) - can be used to view failed pipelines and cancelled stages
   - If a pipeline fails a stage, your pipeline stops and you can view info in the console
   - If pipeline can't perform an action, make sure the IAM Service Role attached has enough permissions
- Used to deploy code from github to beanstalk
- Pipelines - custom is usually best as you get all options
- Pipeline settings is where you name and setup your execution mode for your pipeline
- Source Stage is where you connect to source like Github, setup the branch, and the event filters in which to trigger the pipeline (such as a `push` on a specific branch)
- Build Stage
- Test Stage
- Deploy Stage - this is where you select the elastic beanstalk environment we created
- **NOTE** the role (can click on under settings) that runs the pipeline will need access to ElasticBeanstalk to deploy. You will probably need to add the permission
- Ability to edit the pipeline and add stages (such as DeployToProd)
- **MULTIPLE Action Groups** can be put into a single **STAGE** - Manual approval is good for verifying a change BEFORE pushing to production
- **INVOKE** Action can be used to:
    * Invoke a lambda function within a pipeline
    * Invoke STEP FUNCTIONS such as putting an item in a dynamoDB table or Start a task in ECS Tasks
- Can build a pipeline that runs in different regions
    * **MUST HAVE S3 Artifact Stores in each region where you have actions**
    * Codepipeline must also have read/write access to all artifacts in those regions
    * While **CodeBuild** handles creating multiple templates/artifacts for each region, **CodePipeline** handles copying the artifact to the specific regions

### Events vs Webhooks vs Polling
- Events are preferred when an event like a checkin/push/etc occurs
- Code pipeline exposes a webhook and a script can send information to it
- Code pipeline can also PULL on regular checks
- Actions for stages - 
    * Source - S3, CodeCommit Github
    * Build - Codebuild, Jenkins
    * Test - DeviceFarm, CodeBuild, Jenkins
    * Approval - Manual
    * Deploy - S3, Cloudformation, CodeDeploy, Elastic Beanstalk, ECS, Service Catalog, Alexa Skills
    * Invoke - Lambda, Step Functions

### Best Practices
- One pipeline, one deploy, to multiple deployment GROUPS
- Parallel actions using **RunOrder**
- Deploy to pre-prod, manual approval BEFORE deploying to prod

---

## CodeBuild
- Source can come from CodeCommit, S3, Bitbucket, GitHub
- **Build Instructions**: In code file **buildspec.yml** or insert manually in Console
- Output logs can be stored in S3 and CLoudWatch logs
- Use CloudWatch metrics to monitor build statistics
- Use EventBridge to detect failed builds and trigger notifications
- PRE BUILT IMAGE READY for Java, Ruby, Python, Go, Node.js, Android, .NET Core, PHP
    - Docker - extend to any environment/language you want
- `source code` + `buildspec.yml` IN GITHUB
- CodeBuild - creates a container based on the `buildspec.yml`
- **OPTIONAL OPTIMIZATION**: can store reusable pieces in an S3 bucket (cache)
- Build artifacts stored in an S3 bucket
- `buildspec.yml`
    - Must be at the ROOT of your code
    - `env` define environment variables
        - Variables (plaintext)
        - parameter-store - variables stored in the SSM Parameter Store
        - secrets-manager - variables stored in AWS Secrets Manager
    - `phases` specify `commands` to run
        - `install` - build dependencies
        - `pre_build` - final commands to execute before build
        - `build` - actual build commands
        - `post_build` - finishing touches (i.e. zip output)
    - `artifacts` - what to upload to S3 (encrypted with KMS)
    - `cache` - files to cache (usually dependencies) to S3 for future build speed up
- **Local Build**
    - Can run locally on desktop (after installing Docker) for deep troubleshooting beyond logs
    - You need to use the `CodeBuild Agent`
- **VPC - Virtual Private Cloud**
    - Default, CodeBuild containers are launched outside VPC and cannot access resources INSIDE
    - Can specify a VPC configuration such as VPC ID, Subnet IDs, Security Group IDs
    - After, build can access resources in VPC such as RDS, ElasticCache, EC2, ALB
- **Env Variables**
    - Default vars - `AWS_DEFAULT_REGION`, `CODEBUILD_BUILD_ID`, etc
    - CUSTOM Env Vars 
        - *Static* - defined at build time (i.e. `ENVIRONMENT=production`)
        - *Dynamic* - using SSM parameter store and Secrets Manager (i.e. `SECRET_TOKEN=my_secret_token`)
- **Security**
    - CodeBuild Service Role - access to AWS resources on your behalf
        - Download code from CodeCommit
        - Fetch params from SSM parameter store
        - Fetch secrets from Secrets Manager
        - Upload build artifacts to S3
        - Store logs in Cloudwatch logs
    - In-transit and at rest data encryption
    - Build output artifact encryption
- **Misc**
    - *Build Badges*: Dynamically generated and display latest status of build (at branch level). Accessible via public URL
    - *Triggers*: Commit Event (GitHub) -> Webhook -> EventBridge -> [Lambda] -> Codebuild
    - *Test Reports*: Able to view `Report Group` (Test report) of how tests are running in your build
        - Can also provide custom reports under `reports: files: - "reports/php/*.xml"` within your `buildspec.yml`

---

## CodeDeploy

- Deployment service that automates application deployment
- Deploy to EC2/ECS services, on-premise servers, or Lambda functions
- Automated rollback in case of failed deployments or trigger CloudWatch alarm
- `appspec.yml` defines how the deployment happens
- all at once, half at a time, one at a time - different approaches for updating instances to vary downtimes
- **Blue-Green** - create an alternative auto scaling group with the newest version, load balancer switches to point from v1 to v2
- **CodeDeploy Agent** 
    - must be installed on the EC2 instances as a _pre-req_
    - Must have sufficient permissions to access AMazon S3 to get deployment bundles

### EC2 Deep Dive
- In-Place Deployment
    - Replaces existing instances with v2 from v1
    - Options:
        - Use EC2 tags or ASG to identify specific instances you want to deploy to
        - If using **Auto Scaling Group (ASG)** CodeDeploy will ALSO automatically deploy to any newly launched instances in the ASG
    - Hooks
        - BeforeBlockTraffic
        - AfterBlockTraffic
        - ApplicationStop
        - BeforeInstall
        - AfterInstall
        - ApplicationStart - need this to tell CodeDeploy how to start your application
        - ValidateService
        - BeforeAllowTraffic
        - AfterAllowTraffic
- Blue/Green Deployments
    - **Load Balancer is REQUIRED**
    - Manual Deployment - must use tags to identify
    - Automatic mode - new ASG is provisioned by CodeDeploy, coping settings from v1
    - Same hooks as with manual are run on the `v2` instances EXCEPT the following are ran on `v1` only (as rolling from v1 instances to v2 instances):
        - BeforeBlockTraffic
        - AfterBlockTraffic
- Configurations
    - AllAtOnce
    - HalfAtATime
    - OneAtATime


### Lambda Platform Example
- Can use CodeDeploy with Lambda to shift traffic from v1 to v2
- Linear
    - LambdaLinear10PercentEvery3Minutes -
    - LambdaLinear10PercentEvery10Minutes -
- Canary
    - LambdaCanary10Percent5Minutes - 5 minutes and then all at once after
    - LambdaCanary10Percent30Minutes - 30 minutes and then all at once after
- AllAtOnce - don't have time to test a rollout

### ECS Platform
- Can use CodeDeploy to automate deployment of a new ECS Task Definition to an ECS Service
- **CAN ONLY USE FOR BLUE-GREEN DEPLOYMENTS**
- **LOAD BALANCER is REQUIRED**
- ECS Task Definition (`TaskDefinition`), Load blanacer (`LoadBalancerInfo`) and new container images must be already created
    - These are specified in the `appspec.yml`
- Linear, Canary, and AllAtOnce as with Lambda
 

---

## EventBridge
- CodePipeline (and other AWS tools) send events to EventBridge
- If a pipeline fails, that event in EventBridge COULD:
    * Invoke a Lambda function to diagnose the issue/code
    * Trigger an SNS to notify users of the failure

---

## CloudFormation
- Target for CodePipeline
- Can be used to deploy any kind of AWS resources OR Lambda functions using CDK or SAM (alternative to CodeDeploy)
- CREATE CHANGE SET in CloudFormation -> Manual Approval -> CloudFormation EXECUTE CHANGE SET
- Action Modes for CloudFormation
    * Manual Approval    
        * CREATE or REPLACE a CHANGE SET
        * EXECUTE a CHANGE SET
    * No Manual approval:
        * CREATE or UPDATE a STACK
        * DELETE a STACK
        * REPLACE a FAILED STACK
