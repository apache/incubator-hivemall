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

This page introduces how to run Hivemall on Docker.

<!-- toc -->

> #### Caution
> This docker image contains a single-node Hadoop enviroment for evaluating Hivemall. Not suited for production uses.

# Requirements

 * Docker Engine 1.6+
 * Docker Compose 1.10+

# 1. Build image

## Build using docker-compose
  
  `docker-compose -f resources/docker/docker-compose.yml build`

## Build using docker command
  
  `docker build -f resources/docker/Dockerfile .`

> #### Note
> You can [skip](./getting_started.html#running-pre-built-docker-image-in-dockerhub) building images by using existing Docker images.

# 2. Run container

## Run by docker-compose

  1. Edit `resources/docker/docker-compose.yml`
  2. `docker-compose -f resources/docker/docker-compose.yml up -d && docker attach hivemall`

## Run by docker command

  1. Find a local docker image by `docker images`.
  2. Run `docker run -it ${docker_image_id}`. 
     Refer [Docker reference](https://docs.docker.com/engine/reference/run/) for the command detail.

## Running pre-built Docker image in Dockerhub

  1. Check [the latest tag](https://hub.docker.com/r/hivemall/latest/tags/) first.
  2. Pull pre-build docker image from Dockerhub `docker pull hivemall/latest:20170517`
  3. `docker run -p 8088:8088 -p 50070:50070 -p 19888:19888 -it hivemall/latest:20170517`

You can find pre-built Hivemall docker images in [this repository](https://hub.docker.com/r/hivemall/latest/).

# 3. Run Hivemall on Docker

  1. Type `hive` to run (`.hiverc` automatically loads Hivemall functions)
  2. Try your Hivemall queries!

## Accessing Hadoop management GUIs

* YARN http://localhost:8088/
* HDFS http://localhost:50070/
* MR jobhistory server http://localhost:19888/

Note that you need to expose local ports e.g., by `-p 8088:8088 -p 50070:50070 -p 19888:19888` on running docker image.

## Load data into HDFS (optional)

  You can find an example script to load data into HDFS in `./bin/prepare_iris.sh`.
  The script loads iris dataset into `iris` database.

## Build Hivemall (optional)

  In the container, Hivemall resource is stored in `$HIVEMALL_PATH`.
  You can build Hivemall package by `cd $HIVEMALL_PATH && ./bin/build.sh`.
