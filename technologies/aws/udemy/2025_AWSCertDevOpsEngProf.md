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
- 

## DVA - CodePipeline - Overview (Alternative to CIRCLECI)
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


## Elastic Beanstalk
- Used to create an APPLICATION which then has ENVIRONMENTS that are for that APPLICATION
- ENVIRONMENTS can be a WEB SERVER for a website/web app, or web API, or a worker app that does long-running workloads on demand/schedule
- Environment platforms can be .NET, Docker, PHP, Python, Node.js, Ruby, Java, and a few others

## CodePipeline
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