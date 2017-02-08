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

* Spark v2.0 or later
* Java 7 or later
* hivemall-spark-xxx-with-dependencies.jar
* [define-all.spark](https://github.com/apache/incubator-hivemall/blob/master/resources/ddl/define-all.spark)
* [import-packages.spark](https://github.com/apache/incubator-hivemall/blob/master/resources/ddl/import-packages.spark)

Installation
============

First, you download a compiled Spark package from [the Spark official web page](http://spark.apache.org/downloads.html) and
invoke spark-shell with a compiled Hivemall binary.

```
$ ./bin/spark-shell --jars hivemall-spark-xxx-with-dependencies.jar
```

> #### Notice
> If you would like to try Hivemall functions on the latest release of Spark, you just say `bin/spark-shell` in a Hivemall package.
> This command automatically downloads the latest Spark version, compiles Hivemall for the version, and invokes spark-shell with the compiled Hivemall binary.

Then, you load scripts for Hivemall functions.

```
scala> :load define-all.spark
scala> :load import-packages.spark
```

