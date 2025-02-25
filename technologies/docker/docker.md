# Quick Tips

## Start a Docker Image

```
docker run \
    --rm \
    --name=jobmgr \
    --network flink-network \
    --publish 8081:8081 \
    --env FLINK_PROPERTIES="${FLINK_PROPERTIES}" \
    jobmgr &
```

where jobmgr is the image name in docker.desktop

## Login to Docker

`docker exec -it <image id> sh` - image id can be found in docker.desktop
Example: `docker exec -it 9cb6194c78c4c94b12381cd2b32fcf6b376b13177cea8a4d27134eb4f3f3dd44 sh`

## Save a new version of an Image

`docker commit jobmgr jobmgr-v2`

## Flink Docker + Python App

1. Ensure Docker Desktop is installed and running
2. Follow the steps to create a `jobmanager` with [the Flink instructions for docker](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/deployment/resource-providers/standalone/docker/)
   1. `FLINK_PROPERTIES="jobmanager.rpc.address: jobmanager"`
   2. `docker network create flink-network`
   3. `docker run --rm --name=jobmanager --network flink-network --publish 8081:8081 --env FLINK_PROPERTIES="${FLINK_PROPERTIES}" flink:1.20.0-scala_2.12 jobmanager`
3. Use the Docker Desktop to get the container ID
4. Login to docker using sh: `docker exec -it <container id> bash`
5. Create directories for messages and processed files (did this in `/opt/flink/<project_name>` with `mkdir messages` and `mkdir processed`)
6. Upgrade `apt-get` by running: `apt update && apt upgrade -y`
7. Install Java, Dev Packages, and Source to ensure you get the header files:
   - `apt-get install openjdk-8-jdk`
   - `apt-get install openjdk-8-source`
   - `apt-get install openjdk-8-jdk-headless`
   - See the list of available version with: `apt-cache search openjdk`. Be sure it's the actual JDK / Development Kit. Once installed, verify that `javac` runs in the command line.
8. Update Java home to the newly installed Java directory. You can look in the `/usr/lib/jvm/` folder for your specific install. Example: `export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-arm64"`
9. Install Python: `apt-get install python3`
10. Create a symbolic link from `python` to `python3` as the JVM will attempt to use `python` by default, causing all python Flink jobs to fail: `ln -s /usr/bin/python3 /usr/bin/python`
11. Install Pip: `apt-get install pip`
12. Install Apache PyFlink: `pip install apache-flink`
13. Install New Relic libraries: `pip install newrelic`
14. _OPTIONAL:_ Spin up a NewRelic docker container to monitor Docker containers in another terminal/iterm window: `docker run --detach --name newrelic-infra --network=host --cap-add=SYS_PTRACE  --privileged --pid=host --volume "/:/host:ro" --volume "/var/run/docker.sock:/var/run/docker.sock" --volume "newrelic-infra:/etc/newrelic-infra" --env NRIA_LICENSE_KEY=<testLicense> newrelic/infrastructure:latest`
15. On your local machine, copy the python script over to docker using: `docker cp <local file> <container id>:<path to where file should go>`
16. _OPTIONAL:_ On your local machine, copy the newrelic.ini file for new relic config: `docker cp newrelic.ini <container id>:<path to where file should go>`
17. On the docker image, run `python <file copied in 13>`
18. Send any CSV file to the `messages` folder on docker: `docker cp <local csv file> <container id>:<path to messages folder>/messages/`

## Questions:

- Would you ever build your docker image in the same directory that is in source control or just keep the dockerfile there?
  - You just keep the docker file (that's the artifact) you transfer from dev machine to dev machine then, once you get into CI you publish images out of that automated process into some image registry with auto-incrementing tags
  - Build only stores image data in your docker host data directories. It should have no impact on the files within your repo UNLESS you mount a volume to a running container
