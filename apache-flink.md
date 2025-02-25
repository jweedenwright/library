# Apache Flink

### Challenges

- Streaming data is UNBOUNDED
- Repeatability of results - different results
- Results can expire
- State management
- Latency
- Resource requirements - hardware

## Why Flink when you have Spark?

- Spark replaced Hadoop (Faster)
- Flink replaces Spark for processing stream data
  - Spark was not built for real time (it is NEAR real time) - more batch processing
- Spark - processes data in Batches
  - Manually memory management (Can frequently run out of memory)
- Flink - Uses windows (for time periods) and checkpointing (notate where stream has left off processing, and start back up on that point)
  - Has built-in efficient memory management, more performant
- Kafka could be used to generate a datastream

- [Udemy Training](https://fortyau.udemy.com/course/apache-flink-a-real-time-hands-on-course-on-flink/learn/lecture/11927592#overview)

> A Flink APPLICATION is a Java / Python program that submits/creates one or more Flink jobs
> A Flink JOB is an INSTANCE of the APPLICATION (i.e. the runtime representation) of a logical graph/dataflow graph (job is submitted via calling the `execute()` command)

## [Flink Architecture](https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/flink-architecture/)

## Overview

| Streaming, State, Time, and the use of state snapshots for fault tolerance and failure recovery

- [PyFlink Documentation](https://nightlies.apache.org/flink/flink-docs-release-1.15/api/python/index.html)
  - [PyFlink Datastream Tutorial](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/python/datastream_tutorial/)
  - [PyFlink Table Tutorial](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/python/table_api_tutorial/)
- Real-Time Streaming Pipeline
- Distributed stream-processing framework
- Open Source
- Array of connectors
- Event time processing
- Exactly ONCE consistency
- Fast and Scalable
- Very low latency
- High availability
- _Connectors_: HDFS, Kafka, Amazon Kinesis, RabbitMQ, Google PubSub, Cassandra
- Can rate/accumulate events based on TIMESTAMPS
- _Java and IntelliJ_
- Competitors - Apache Spark, Samza, Storm, and others
- SOURCE (File/Kafka, Flume, Socket) -> OPERATIONS -> SINK (HDFs, Database, Memory, Log)

## Running Flink

- _START CLUSTER_: In the folder where you've extracted the Flink source (and after installing Java), you can run: `./<flink folder>/bin/start-cluster.sh`
- _STOP CLUSTER_: In the folder where you've extracted the Flink source (and after installing Java), you can run: `./flink-1.20.0/bin/stop-cluster.sh`

## Technical Details

### Setup

- Spin up a new virtual python environment using: `py -m venv .` (example: `py -m venv .myenv`)
- To activate and join the environment, use: `source ./bin/activate` (example: `source .myenv/bin/activate`)
- To stop the environment and leave, use: `deactivate`
- [Configuring python execution environment](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/python/python_config/)

### DataStream APIs

- High-level stream-processing API used by Flink
- _Sources_: Message queues, Socket streams, Files
- Creates and uses _DataStream_ objects for unbounded data
- _DataStreams_ are the basis for Flink Pipelines
  - _DataStream_ objects are created, transformed, aggregated and published
  - Map, flat map, filter, reduce, and aggregation operations can be performed on _Datastreams_
  - Immutable, so unchangeable once created
  - Cannot inspect elements inside, must use `DataStream` API operations to explore (i.e. TRANSFORMATIONS)
- Timestamps and source events for windowing and aggregations
- Real-time state management capabilities

### Connectors (EXTRACT | IMPORT)

- Create from a list object:
  - `env.from_collection(collection=[(1,'a|b'),(2,'bb|a'),(3,'aaa|a')],...`
- Create from a connector using `add_source`:
  - `env.add_jars("file:///path/to/connector.jar")`: for when you need to add libraries/connectors
  - Add the logic for your specific connector: `consumer = FlinkKafkaConsumer(...` (example is for Kafka)
  - Add the source: `ds = env.add_source(consumer)`
- Create from a connector using `from_source`:
  - Currently only supports `NumberSequenceSource` and `FileSource`
  - Can be used for BOTH `batch` and `streaming` executing modes
- Create using table and SQL connectors:
  - Setup the table environment (using the execution environment we first start with)
    - `t_env = StreamTableEnvironment.create(stream_execution_environment=env)`

```
t_env.execute_sql("""
    CREATE TABLE my_source (
        a INT,
        b VARCHAR
    ) WITH (
        'connector' = 'datagen',
        'number-of-rows' = '10'
    )
""")

ds = t_env.to_append_stream(
t_env.from_path('my_source'),
Types.ROW([Types.INT(), Types.STRING()]))
```

### DataStream Transformations (TRANSFORM | ENRICHMENT)

- _Operators_ transform one or MORE `DataStream` into a new `DataStream`
  - Simple example: `ds = ds.map(lambda a: a + 1)`
- _Map_:
- [`data_stream.flat_map`](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/overview/#flatmap): take one element and produces zero, one, or more elements
- [`data_stream.filter`](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/overview/#filter): Retain elements in Datastream where function returns true
- [`data_stream.keyBy`](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/overview/#keyby): Equivalent to SQL's `groupBy`. Logically partitions a stream into disjoint partitions (same key = same partition). Can be EXPENSIVE since it may have to shuffle every keyby to a different node (lots of network communication)
- [`data_stream.window`](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/overview/#window): Requires a keyed stream
  - Tumbling Windows `TumblingEventTimeWindows` - group events into some rolling time interval (such as events grouped into rolling 5 min intervals)
  - Sliding Windows `SlidingEventTimeWindows` - window of a specific interval (such as 10 mins), but can specify how frequently a NEW window is started. So a 10 min interval with 5 min sliding means a new window would be created every 5 mins. Events that arrive during this time are assigned to multiple windows.
  - Session Windows `EventTimeSessionWindows` - a session window does NOT overlap and do not have fixed start and end times. Sessions open on receipt of an element and close after a certain period of time when no elements arrive. _Example: KeyBy a unique ID (such as a Google or Adobe Id), and then go into a session window_
  - Global Windows `GlobalWindows` - assign _ALL_ elements with the same key to the same _global_ window.
    - _Only useful if a custom trigger is used otherwise no computation will be performed (as this window does NOT have a natural end)_
- [`data_stream.window_all`](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/overview/#windowall)

### JobManager

- Responsible for coordinating the distributed execution of Flink Applications
- Decides when to schedule the next task/set of tasks
- Reacts to finished tasks or failures
- Coordinates checkpoints (_DEPENDENCIES??_)
- Coordinates recovery on failures
- _Resource Manager_
  - Resource de-/allocation and provisioning in a Flink cluster
  - Manages _task slots (a.k.a TaskManagers)_
  - In standalone, can only manage existing slots _NOT START NEW TaskManagers_
- _Dispatcher_
  - REST interface to submit Flink applications

### [File System / Source](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/connectors/datastream/filesystem/)

- A bounded File Source lists all files (via SplitEnumerator - a recursive directory list with filtered-out hidden files) and reads them all.
- An unbounded File Source is created when configuring the enumerator for periodic file discovery. In this case, the SplitEnumerator will enumerate like the bounded case but, after a certain interval, repeats the enumeration. For any repeated enumeration, the SplitEnumerator filters out previously detected files and only sends new ones to the SourceReader.

### [Apache Flink: Real-Time Data Engineering | Topics Covered](https://www.linkedin.com/learning/apache-flink-real-time-data-engineering/related-prerequisite-courses?autoSkip=true&resume=false&u=0)

- _May want to take_ - Apache Flink: Batch Mode Data Engineering Course
- KeyBy for partitioning streams
- Windowing for aggregations
- Splitting and merging streams
- Event time processing
- Stateful processing
- Streaming sources and syncs

## [Apache Flink Python DataStream](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/python/datastream_tutorial/)

### Env Setup

- Python program must first declare execution environment `StreamExecutionEnvironment`, i.e. the context in which the streaming program is executed.
- This is what you'll use to set the properties of your job (parallelism, restart strategy), create your sources, and finally trigger the execution of the job

```
env = StreamExecutionEnvironment.get_execution_environment()
env.set_runtime_mode(RuntimeExecutionMode.BATCH)
env.set_parallelism(1)
```

### Sources

- Sources ingest data from external systems, such as Apache Kafka, Rabbit MQ, or Apache Pulsar, _INTO_ Flink Jobs
- Simple example to read data from a file:

```
ds = env.from_source(
    source=FileSource.for_record_stream_format(StreamFormat.text_line_format(),
                                               input_path)
                     .process_static_file_set().build(),
    watermark_strategy=WatermarkStrategy.for_monotonous_timestamps(),
    source_name="file_source"
)
```

### Example Walkthrough of how Flink Works

- Count the number of works that start with N
- First block of data comes in:
  - READ THE FILE -> STORED ON _NODE A1_
  - FILTER OPERATOR -> RESULTS STORED ON _NODE A2_
  - GROUP BY OPERATION -> RESULTS STORED ON _NODE A3_
  - SUM / COUNT -> OUTPUT
- In Hadoop, all of these intermittent outputs go to FILES on the system, but next node is able to process data from disk if the process fails
- _In Flink_ - Intermittent output is STORED IN THE NODE'S CURRENT MEMORY
  - If Flink transform fails, Flink will spin up a new node to do the processing that failed to move the process on
  - _Each Dataset object is stored as a NODE in Flink and can be recreated if one fails_
  - Rules:
    - Dataset are IMMUTABLE (cannot be changed)
    - Only 1 operation per dataset
    - Each dataset contains list of dependencies in order to get to IT'S OPERATION TO PERFORM (i.e. NODE 3 needs NODE 1 and NODE 2 output)
