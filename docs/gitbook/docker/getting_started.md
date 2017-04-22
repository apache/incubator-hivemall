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

# Getting Started

## How to run
1. Install the following
  * `Docker Engine 1.6+`
  * [OPT] `Docker Compose 1.10+` (optional but recommended)
2. Build image
  * [RECOMMENDED] `docker-compose -f resources/docker/docker-compose.yml build`
  * Or `docker build -f resources/docker/Dockerfile .`
3. Run container
  * [RECOMMENDED]
    1. Edit `resources/docker/docker-compose.yml`
    2. `docker-compose -f resources/docker/docker-compose.yml up -d && docker attach hivemall`
  * Or `docker run -it ${your data volume and port options} hivemall`
4. Start Hivemall (in container)
  1. [OPT] Load data into HDFS
    * You can load iris dataset by just `./prepare_iris.sh`
  2. Build Hivemall
   * You can build Hivemall either in or out of container
   * If in container, `cd /hivemall && mvn package -Dmaven.test.skip=true -pl core`
  3. `hive` to run Hive with Hivemall
  4. Try your queries!


## Notice
* **Use for testing only**
* Including build tools for Hivemall
