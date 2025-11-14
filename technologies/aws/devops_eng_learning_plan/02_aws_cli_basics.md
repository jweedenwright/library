# AWS Command Line Interface (CLI) Basics

## Overview
* unified tool to manage your AWS services
* `brew install awscli` or `pip install awscli`

```
aws configure
AWS Access Key ID [None]: AK#####
AWS Secret Access Key [None]: #########00027
Default region name [None]: us-east-1
Default output format [None]: json

-- see instances in a table
aws ec2 describe-instances --output table
-- see a list of all instance ids
aws ec2 describe-instances --query 'Reservations[].Instances[].InstanceId'

```