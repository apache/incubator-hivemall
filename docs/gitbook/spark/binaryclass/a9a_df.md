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
val rawTrainDf = spark.read.format("libsvm").load("a9a")

val (max, min) = rawTrainDf.select(max($"label"), min($"label")).collect.map {
  case Row(max: Double, min: Double) => (max, min)
}

val trainDf = rawTrainDf.select(
    // `label` must be [0.0, 1.0]
    rescale($"label", lit(min), lit(max)).as("label"),
    $"features"
  )

scala> trainDf.printSchema
root
 |-- label: float (nullable = true)
 |-- features: vector (nullable = true)

scala> :paste
val testDf = spark.read.format("libsvm").load("a9a.t")
  .select(rowid(), rescale($"label", lit(min), lit(max)).as("label"), $"features")
  .explode_vector($"features")
  .select($"rowid", $"label".as("target"), $"feature", $"weight".as("value"))
  .cache

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
val modelDf = trainDf
  .train_logregr(append_bias($"features"), $"label")
  .groupBy("feature").avg("weight")
  .toDF("feature", "weight")
  .cache
```

#Test

```scala
scala> :paste
val predictDf = testDf
  .join(modelDf, testDf("feature") === modelDf("feature"), "LEFT_OUTER")
  .select($"rowid", ($"weight" * $"value").as("value"))
  .groupBy("rowid").sum("value")
  .select(
    $"rowid",
    when(sigmoid($"sum(value)") > 0.5, 1.0).otherwise(0.0).as("predicted")
  )
```

#Evaluation

```scala
scala> val df = predictDf.join(testDf, predictDf("rowid").as("id") === testDf("rowid"), "INNER")

scala> (df.where($"target" === $"predicted").count + 0.0) / df.count
Double = 0.8327921286841418
```

