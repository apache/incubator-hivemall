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

# 2. Run container

## Run by docker-compose

  1. Edit `resources/docker/docker-compose.yml`
  2. `docker-compose -f resources/docker/docker-compose.yml up -d && docker attach hivemall`

## Run by docker command

  1. Find a local docker image by `docker images`.
  2. Run `docker run -it ${docker_image_id}`. 
     Refer [Docker reference](https://docs.docker.com/engine/reference/run/) for the command detail.

# 3. Build Hivemall

  In the container, Hivemall resource is stored in `$HIVEMALL_PATH`.
  You can build Hivemall package by `cd $HIVEMALL_PATH && ./bin/build.sh`.

# 4. Run Hivemall on Docker

  1. Type `hive` to run (see `.hiverc` loads Hivemall functions)
  2. Try your Hivemall queries!

## Load data into HDFS (optional)

  You can find an example script to load data into HDFS in `./bin/prepare_iris.sh`.
  The script loads iris dataset into `iris` database.
