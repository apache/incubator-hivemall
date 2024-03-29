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

* Spark v2.2 or later
* Java 8 or later
* `hivemall-all-<version>.jar` that can be found in [Maven central](https://search.maven.org/search?q=a:hivemall-all) (or use packages built by `bin/build.sh`).
* [define-all.spark](https://github.com/apache/incubator-hivemall/blob/master/resources/ddl/define-all.spark)

Installation
============

First, you download a compiled Spark package from [the Spark official web page](https://spark.apache.org/downloads.html) and invoke spark-shell with a compiled Hivemall binary.

```
$ spark-shell --jars target/hivemall-all-<version>-incubating-SNAPSHOT.jar
```

Installation via [Spark Packages](https://spark-packages.org/package/apache-hivemall/apache-hivemall)
============

In another way to install Hivemall, you can use a `--packages` option.

```
$ spark-shell --packages org.apache.hivemall:hivemall-all:<version>
```

You find available Hivemall versions on [Maven repository](https://mvnrepository.com/artifact/org.apache.hivemall/hivemall-all/).


Then, you load scripts for Hivemall functions.

```
scala> :load ~/workspace/incubator-hivemall/resources/ddl/define-all.spark
```