<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Getting started with Hivemall on Docker

This page introduces how to run Hivemall on Docker.

<!-- toc -->

> #### Caution
> This docker image contains a single-node Hadoop enviroment for evaluating Hivemall. Not suited for production uses.

## Requirements

 * Docker Engine 1.6+
 * Docker Compose 1.10+

## Build image

You have two options in order to build a **hivemall** docker image:

### Using `docker-compose`
  
```
$ docker-compose -f resources/docker/docker-compose.yml build
```

### Using `docker` command
  
```
$ docker build -f resources/docker/Dockerfile .
```

> #### Note
> You can [skip](./getting_started.html#running-pre-built-docker-image-in-docker-hub) building images if you try to use a pre-build docker image from Docker Hub. However, since the Docker Hub repository is experimental one, the distributed image is NOT built on the "latest" commit in [our master branch](https://github.com/apache/incubator-hivemall).

## Run container

If you built an image by yourself, it can be launched by either `docker-compose` or `docker` command:

### By `docker-compose`

```
$ docker-compose -f resources/docker/docker-compose.yml up -d && docker attach hivemall
```

You can edit `resources/docker/docker-compose.yml` as needed.

For example, setting `volumes` options enables to mount your local directories to the container as follows:

```yml
volumes:
  - "../../:/opt/hivemall/" # mount current hivemall dir to `/opt/hivemall` ($HIVEMALL_PATH) on the container
  - "/path/to/data/:/root/data/" # mount resources to container-side  `/root/data` directory
```

### By `docker` command

Find a local docker image by `docker images`, and hit:

```
$ docker run -p 8088:8088 -p 50070:50070 -p 19888:19888 -it ${docker_image_id}
```

Refer [Docker reference](https://docs.docker.com/engine/reference/run/) for the command detail.

Similarly to the `volumes` option in the `docker-compose` file, `docker run` has `--volume` (`-v`) option: 

```
$ docker run ... -v /path/to/local/hivemall:/opt/hivemall
```

### Running pre-built Docker image in Docker Hub

> #### Caution
> This part is experimental. Hivemall in the pre-built image might be out-of-date compared to the latest version in [our master branch](https://github.com/apache/incubator-hivemall).

You can find pre-built Hivemall docker images in [this repository](https://hub.docker.com/r/hivemall/latest/).

1. Check [the latest tag](https://hub.docker.com/r/hivemall/latest/tags/) first
2. Pull pre-build docker image from Docker Hub: 
```
$ docker pull hivemall/latest:20170517
```
3. Launch the pre-build image:
```
$ docker run -p 8088:8088 -p 50070:50070 -p 19888:19888 -it hivemall/latest:20170517
```

## Run Hivemall on Docker

  1. Type `hive` to run (`.hiverc` automatically loads Hivemall functions)
  2. Try your Hivemall queries!

### Accessing Hadoop management GUIs

* YARN http://localhost:8088/
* HDFS http://localhost:50070/
* MR jobhistory server http://localhost:19888/

Note that you need to expose local ports e.g., by `-p 8088:8088 -p 50070:50070 -p 19888:19888` on running docker image.

### Load data into HDFS (optional)

You can find an example script to load data into HDFS in `$HOME/bin/prepare_iris.sh`.
  The script loads iris dataset into `iris` database:
  
```
# cd $HOME && ./bin/prepare_iris.sh
```

```
# hive
hive> use iris;
hive> select * from iris_raw limit 5;
OK
1       Iris-setosa     [5.1,3.5,1.4,0.2]
2       Iris-setosa     [4.9,3.0,1.4,0.2]
3       Iris-setosa     [4.7,3.2,1.3,0.2]
4       Iris-setosa     [4.6,3.1,1.5,0.2]
5       Iris-setosa     [5.0,3.6,1.4,0.2]
```

Once you prepared the `iris` database, you are ready to move on to [our multi-class classification tutorial](../multiclass/iris_dataset.html).

### Build Hivemall (optional)

In the container, Hivemall resource is stored in `$HIVEMALL_PATH`.
You can build Hivemall package by `cd $HIVEMALL_PATH && ./bin/build.sh`.
