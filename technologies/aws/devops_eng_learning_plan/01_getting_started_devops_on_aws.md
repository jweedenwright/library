# [DevOps Engineer Learning Plan (includes Labs)](https://skillbuilder.aws/learning-plan/FMP175FCDT/devops-engineer-learning-plan-includes-labs/P8E3Q12Q6H)

##  What is DevOps (Development and Operations)
- Combination of cultural philosophies, practices, and tools
    - Removing barriers and sharing end-to-end responsibility
    - Processes for speed and quality, that streamline people's work
    - Tools that align with processes and automate repeatable tasks (i.e. more efficient and reliable)
- Increases orgs ability to deliver applications and services at high velocity
- **Dev** - People and processes that create software. Change, release quickly
- **Ops** - Teams and Processes that deliver and monitor the software. Stability of the application.
- Older processes are often built on monolithic software that is hard to push updates to without an entire new version
- Often older processes also use other development processes, such as waterfall, to deploy over longer intervals and without a lot of flexibility/agility for change

### Goals
- Increase innovation and agility - can change
- Improve quality - tests and builds are automated
- Release faster - tests and builds are automated
- Increase reliability - via continuous integration and delivery ensuring all pieces are functional and safe
- Lower costs
- Scalability - automatically scale up or down based on demand/need
- Security - automatic enforcement rules, fine-grained controls, and configuration management techniques to ensure compliance policies are met

### Culture
- Create a highly collaborative environment - break down silos, align goals, and have end-to-end ownership for the entire team (dev, test, security, ops, and others)
- Repeatable tasks are automated, allowing the team to focus on innovation. An example iks `infrastructure-as-code (IaC)` where predefined/approved environments are used for repeatable, consistent environments
- Consistent feedback loops from customers (customer-first)
- Streamlined processes, clear communication, and monitoring help keep customer satisfaction high
- Develop small and release often bring in agility/responsiveness
- Embed security at every phase and not just at the end
- Continuously experiment, learn, and improve
- *Typical Workflow: Code / Build / Test / Release / Deploy / Monitor*
- 6 Primary Practices
    - *Communication and Collaboration:* transparency of information and communication, collective ownership
    - *Monitoring and Observability:* assess/measure how effective changes to the app/infrastructure are and their impact on performance and experience
    - *Continuous Integration (CI):* regularly merged code to repo where automated builds and tests can be run. This will help detect issues early and can resolve quickly. It also limits how complex merges between different code sets are. Address bugs quicker, improve software quality, and reduce time to validate/release
    - *Continuous Delivery (other envs) / Deployment (PROD) (CD):* all code changes are automatically built, tested, and deployed to a non-prod test/staging env (manual approval necessary for Production)
    - *Microservices Architecture*: applications should be built as a set of loosely coupled services, each focused on a set of capabilities to solve a business problem. No code or services need to be shared with other services. 
    - *Infrastructure as Code (IaC)*: rather than using error prone manual creation of environments (that is also very difficult to scale), this approach uses code and software development techniques to provision and manage infrastructure

### Tools
- Cloud resources
- Development
    - IDE - AWS Cloud9, VS Code
    - SDKs - AWS SDK for Java
    - Source code repositories: ~AWS CodeCommit~, Github
- CI/CD
    - Build Tools: AWS CodeBuild, Jenkins, Databricks, CircleCI
    - Deployment Tools: AWS CodeDeploy, AWS CloudFormation
    - Pipeline Automation Tools: AWS CodePipeline, Jenkins, Databricks, CircleCI
Infrastructure Automation: Programmatically define your infrastructure, including constraints, to repeatedly and consistently provision your environments
    - Infrastructure Automation Tools: AWS CloudFormation, Terraform, AWS Elastic Beanstalk
Containers and Serverless: FOCUS ON APPLICATION, not on the details of the host environment
    - Serverless Services: AWS Lambda, AWS Fargate
    - Container Services: 
        - Runtimes: Docker, Containerd
        - Orchestration: Amazon Elastic Container Service (ECS), Kubernetes, Amazon Elastic Kubernetes Service (Amazon EKS)
Monitoring and Observability
    - AWS X-Ray, Amazon Cloudwatch, AWS Config, AWS CloudTrail
- AWS Tooling
    - INFRASTRUCTURE - AWS CloudFormation - used to create infrastructure and supporting resources for dev environments
    - DEVELOPMENT - AWS Cloud9 to write, run and debug code as well as make changes to code in an ~AWS CodeCommit~ repo
    - CODE - ~AWS CodeCommit - securely store and source control code~ CodeCommit is deprecated and so another source should be used
    - BUILD / TEST - AWS CodeBuild - automatically compile source code, run tests, produce packages ready to deploy, scales to meet your build volume, can run separate builds concurrently
    - DEPLOY - AWS CodeDeploy - fully managed deployment service that automates software deployments to compute services such as Amazon EC2, AWS Fargate, AWS Lambda, or on-prem servers
    - MONITOR 
        - AWS X-Ray - Collect data about requests your application serves and provide tools to view, filter, and gain insights into issues and optimizations
        - Amazon CloudWatch - monitors AWS resources and applications ran on AWS in real time. Can use to collect and track metrics
    - **WORKFLOW / PIPELINE | AWS CodePipeline** - continuous delivery service that helps you automate your release pipelines for fast and reliable application/infrastructure updates. Automates build, test, and deploy of your release process every time there is a code change (based on the release model you define). Will enhance your ability to further integrate with other AWS and third-party products
    - Amazon Elastic Container Service allows you to use docker containers to easily run application on Elastic Compute Cloud (EC2) instances. No need to install, operate, and scale container management software.
    - Amazon EC2 (Elastic Compute Cloud) hosts webpages / is a web service that provides secure, resizable compute capacity in the cloud.
    - Monitoring:
        - AWS X-Ray - distributed tracing system
        - Amazon cloudwatch - data and actionable insights for AWS resources
        - AWS Config - assess, audit, and evaluate configurations for AWS resources
        - AWS CloudTrail - enables governance, compliance, operational auditing, and risk auditing of AWS account

### Demo
- Explain how to use AWS CodeDeploy to deploy web applications to Amazon Elastic Compute Cloud (Amazon EC2) servers in multiple Regions. 
- Explain how to use AWS CodePipeline to establish a continuous integration and perform continuous deployment to multiple Regions.
- Demo pipeline deploys to 2 regions - region 1 is used to test and, when good to go, can pass the manual check and deploy the changes














