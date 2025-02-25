# Kafka Notes

* docker-compose -f kafka-single-node.yml up -d
* Log into docker container: `docker exec -it kafka-broker /bin/bash`
* Navigate to bin directory: `cd /opt/bitnami/kafka/bin`
* Create topic:
```
./kafka-topics.sh \
            --bootstrap-server localhost:29092 \
            --create \
            --topic kafka.medaxion.incoming \
            --partitions 1 \
            --replication-factor 1
```
* Publish to topic:
```
./kafka-console-producer.sh \
            --bootstrap-server localhost:29092 \
            --topic kafka.medaxion.incoming
```
* Consume from topic:
```
./kafka-console-consumer.sh \
            --bootstrap-server localhost:29092 \
            --topic kafka.medaxion.incoming \
            --from-beginning
```