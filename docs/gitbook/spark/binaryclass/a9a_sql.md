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

a9a
===
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a

Data preparation
================

```sh
$ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
$ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t
```

```scala
scala> :paste
park.read.format("libsvm").load("a9a")
  .select($"label", to_hivemall_features($"features").as("features"))
  .createOrReplaceTempView("rawTrainTable")

val (max, min) = sql("SELECT MAX(label), MIN(label) FROM rawTrainTable").collect.map {
  case Row(max: Double, min: Double) => (max, min)
}.head

// `label` must be [0.0, 1.0]
sql(s"""
  CREATE OR REPLACE TEMPORARY VIEW trainTable AS
    SELECT rescale(label, $min, $max) AS label, features
      FROM rawTrainTable
""")

scala> trainDf.printSchema
root
 |-- label: float (nullable = true)
 |-- features: vector (nullable = true)

scala> :paste
spark.read.format("libsvm").load("a9a.t")
  .select($"label", to_hivemall_features($"features").as("features"))
  .createOrReplaceTempView("rawTestTable")

sql(s"""
  CREATE OR REPLACE TEMPORARY VIEW testTable AS
    SELECT
        rowid() AS rowid,
        rescale(label, $min, $max) AS target,
        features
      FROM
        rawTestTable
""")

// Caches data to fix row IDs
sql("CACHE TABLE testTable")

sql("""
  CREATE OR REPLACE TEMPORARY VIEW testTable_exploded AS
    SELECT
        rowid,
        target,
        extract_feature(ft) AS feature,
        extract_weight(ft) AS value
      FROM (
        SELECT
            rowid,
            target,
            explode(features) AS ft
          FROM
            testTable
        )
""")

scala> testDf.printSchema
root
 |-- rowid: string (nullable = true)
 |-- target: float (nullable = true)
 |-- feature: string (nullable = true)
 |-- value: double (nullable = true)
```

Tutorials
================

[Logistic Regression]
---

#Training

```scala
scala> :paste
sql("""
  CREATE OR REPLACE TEMPORARY VIEW modelTable AS
    SELECT
        feature, AVG(weight) AS weight
      FROM (
        SELECT
            train_logistic_regr(add_bias(features), label) AS (feature, weight)
          FROM
            trainTable
          )
      GROUP BY
        feature
""")
```

#Test

```scala
scala> :paste
sql("""
  CREATE OR REPLACE TEMPORARY VIEW predicted AS
    SELECT
        rowid,
        CASE
          WHEN sigmoid(sum(weight * value)) > 0.50 THEN 1.0
          ELSE 0.0
        END AS predicted
      FROM
        testTable_exploded t LEFT OUTER JOIN modelTable m
          ON t.feature = m.feature
      GROUP BY
        rowid
""")
```

#Evaluation

```scala
val num_test_instances = spark.table("testTable").count

sql(s"""
  SELECT
      count(1) / $num_test_instances AS eval
    FROM
      predicted p INNER JOIN testTable t
        ON p.rowid = t.rowid
    WHERE
      p.predicted = t.target
""")

+------------------+
|              eval|
+------------------+
|0.8327921286841418|
+------------------+
```

