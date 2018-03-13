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
        
Prerequisites
============

* Hadoop v2.4.0 or later
* Hive v0.13 or later
* Java 7 or later
* [hivemall-all-xxx.jar](https://search.maven.org/#search%7Cga%7C1%7Ca%3A%22hivemall-all%22)
* [define-all.hive](https://github.com/apache/incubator-hivemall/blob/master/resources/ddl/define-all.hive) (of a given version, e.g., [v0.5.0](https://github.com/apache/incubator-hivemall/blob/v0.5.0/resources/ddl/define-all.hive))

> #### Note
> 

Installation
============

Add the following two lines to your `$HOME/.hiverc` file.

```
add jar /home/myui/tmp/hivemall-all-xxx.jar;
source /home/myui/tmp/define-all.hive;
```

This automatically loads all Hivemall functions every time you start a Hive session. Alternatively, you can run the following command each time.

```
$ hive
add jar /tmp/hivemall-all-xxx.jar;
source /tmp/define-all.hive;
```


Other choices
=============

You can also run Hivemall on the following platforms:

* [Apache Spark](../spark/getting_started/installation.md)
* [Apache Pig](https://github.com/daijyc/hivemall/wiki/PigHome)
* [Apache Hive on Docker](../docker/getting_started.md) for testing


Build from Source
==================

```sh
$ git clone https://github.com/apache/incubator-hivemall.git
$ cd incubator-hivemall
$ bin/build.sh
```

Then, you can find Hivemall jars in `./target`.
