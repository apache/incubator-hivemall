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

`leftDf.top_k_join(k: Column, rightDf: DataFrame, joinExprs: Column, score: Column)` only joins the top-k records of `rightDf` for each `leftDf` record with a join condition `joinExprs`. An output schema of this operation is the joined schema of `leftDf` and `rightDf` plus (rank: Int, score: `score` type).

`top_k_join` is much IO-efficient as compared to regular joining + ranking operations because `top_k_join` drops unsatisfied records and writes only top-k records to disks during joins.

> #### Caution
> * `top_k_join` is supported in the DataFrame of Spark v2.1.0 or later.
> * A type of `score` must be ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, or DecimalType.
> * If `k` is less than 0, the order is reverse and `top_k_join` joins the tail-K records of `rightDf`.

# Usage

For example, we have two tables below;

- An input table (`leftDf`)

```scala
scala> :paste
val leftDf = Seq(
  (1, "b", 0.3, 0.3),
  (2, "a", 0.5, 0.4),
  (3, "a", 0.1, 0.8),
  (4, "c", 0.2, 0.2),
  (5, "a", 0.1, 0.4),
  (6, "b", 0.8, 0.8)
).toDF("userId", "group", "x", "y")

scala> leftDf.show
+------+-----+---+---+
|userId|group|  x|  y|
+------+-----+---+---+
|     1|    b|0.3|0.3|
|     2|    a|0.5|0.4|
|     3|    a|0.1|0.8|
|     4|    c|0.2|0.2|
|     5|    a|0.1|0.4|
|     6|    b|0.8|0.8|
+------+-----+---+---+
```

- A reference table (`rightDf`)

```scala
scala> :paste
val rightDf = Seq(
  ("a", "pos1", 0.0, 0.1),
  ("a", "pos2", 0.9, 0.3),
  ("a", "pos3", 0.3, 0.2),
  ("b", "pos4", 0.5, 0.7),
  ("b", "pos5", 0.4, 0.2),
  ("c", "pos6", 0.8, 0.7),
  ("c", "pos7", 0.3, 0.3),
  ("c", "pos8", 0.4, 0.2),
  ("c", "pos9", 0.3, 0.8)
).toDF("group", "position", "x", "y")

scala> rightDf.show
+-----+--------+---+---+
|group|position|  x|  y|
+-----+--------+---+---+
|    a|    pos1|0.0|0.1|
|    a|    pos2|0.9|0.3|
|    a|    pos3|0.3|0.2|
|    b|    pos4|0.5|0.7|
|    b|    pos5|0.4|0.2|
|    c|    pos6|0.8|0.7|
|    c|    pos7|0.3|0.3|
|    c|    pos8|0.4|0.2|
|    c|    pos9|0.3|0.8|
+-----+--------+---+---+
```

In the two tables, the example computes the nearest `position` for `userId` in each `group`.
The standard way using DataFrame window functions would be as follows:

```scala
scala> paste:
val computeDistanceFunc =
  sqrt(pow(inputDf("x") - masterDf("x"), lit(2.0)) + pow(inputDf("y") - masterDf("y"), lit(2.0)))

val resultDf = leftDf.join(
    right = rightDf,
    joinExpr = leftDf("group") === rightDf("group")
  )
  .select(inputDf("group"), $"userId", $"posId", computeDistanceFunc.as("score"))
  .withColumn("rank", rank().over(Window.partitionBy($"group", $"userId").orderBy($"score".desc)))
  .where($"rank" <= 1)
```

You can use `top_k_join` as follows:

```scala
scala> paste:
import org.apache.spark.sql.hive.HivemallOps._

val resultDf = leftDf.top_k_join(
    k = lit(-1),
    right = rightDf,
    joinExpr = leftDf("group") === rightDf("group"),
    score = computeDistanceFunc.as("score")
  )
```

The result is as follows:

```scala
scala> resultDf.show
+----+-------------------+------+-----+---+---+-----+--------+---+---+
|rank|              score|userId|group|  x|  y|group|position|  x|  y|
+----+-------------------+------+-----+---+---+-----+--------+---+---+
|   1|0.09999999999999998|     4|    c|0.2|0.2|    c|    pos9|0.3|0.8|
|   1|0.10000000000000003|     1|    b|0.3|0.3|    b|    pos5|0.4|0.2|
|   1|0.30000000000000004|     6|    b|0.8|0.8|    b|    pos4|0.5|0.7|
|   1|                0.2|     2|    a|0.5|0.4|    a|    pos3|0.3|0.2|
|   1|                0.1|     3|    a|0.1|0.8|    a|    pos1|0.0|0.1|
|   1|                0.1|     5|    a|0.1|0.4|    a|    pos1|0.0|0.1|
+----+-------------------+------+-----+---+---+-----+--------+---+---+
```

`top_k_join` is also useful for Spark Vector users.
If you'd like to filter the records having the smallest squared distances between vectors, you can use `top_k_join` as follows;

```scala
scala> import org.apache.spark.ml.linalg._
scala> import org.apache.spark.sql.hive.HivemallOps._
scala> paste:
val leftDf = Seq(
  (1, "a", Vectors.dense(Array(1.0, 0.5, 0.6, 0.2))),
  (2, "b", Vectors.dense(Array(0.2, 0.3, 0.4, 0.1))),
  (3, "a", Vectors.dense(Array(0.8, 0.4, 0.2, 0.6))),
  (4, "a", Vectors.dense(Array(0.2, 0.7, 0.4, 0.8))),
  (5, "c", Vectors.dense(Array(0.4, 0.5, 0.6, 0.2))),
  (6, "c", Vectors.dense(Array(0.3, 0.9, 1.0, 0.1)))
).toDF("userId", "group", "vector")

scala> leftDf.show
+------+-----+-----------------+
|userId|group|           vector|
+------+-----+-----------------+
|     1|    a|[1.0,0.5,0.6,0.2]|
|     2|    b|[0.2,0.3,0.4,0.1]|
|     3|    a|[0.8,0.4,0.2,0.6]|
|     4|    a|[0.2,0.7,0.4,0.8]|
|     5|    c|[0.4,0.5,0.6,0.2]|
|     6|    c|[0.3,0.9,1.0,0.1]|
+------+-----+-----------------+

scala> paste:
val rightDf = Seq(
  ("a", "pos-1", Vectors.dense(Array(0.3, 0.4, 0.3, 0.5))),
  ("a", "pos-2", Vectors.dense(Array(0.9, 0.2, 0.8, 0.3))),
  ("a", "pos-3", Vectors.dense(Array(1.0, 0.0, 0.3, 0.1))),
  ("a", "pos-4", Vectors.dense(Array(0.1, 0.8, 0.5, 0.7))),
  ("b", "pos-5", Vectors.dense(Array(0.3, 0.3, 0.3, 0.8))),
  ("b", "pos-6", Vectors.dense(Array(0.0, 0.7, 0.5, 0.6))),
  ("b", "pos-7", Vectors.dense(Array(0.1, 0.8, 0.4, 0.5))),
  ("c", "pos-8", Vectors.dense(Array(0.8, 0.3, 0.2, 0.1))),
  ("c", "pos-9", Vectors.dense(Array(0.7, 0.5, 0.8, 0.3)))
  ).toDF("group", "position", "vector")

scala> rightDf.show
+-----+--------+-----------------+
|group|position|           vector|
+-----+--------+-----------------+
|    a|   pos-1|[0.3,0.4,0.3,0.5]|
|    a|   pos-2|[0.9,0.2,0.8,0.3]|
|    a|   pos-3|[1.0,0.0,0.3,0.1]|
|    a|   pos-4|[0.1,0.8,0.5,0.7]|
|    b|   pos-5|[0.3,0.3,0.3,0.8]|
|    b|   pos-6|[0.0,0.7,0.5,0.6]|
|    b|   pos-7|[0.1,0.8,0.4,0.5]|
|    c|   pos-8|[0.8,0.3,0.2,0.1]|
|    c|   pos-9|[0.7,0.5,0.8,0.3]|
+-----+--------+-----------------+

scala> paste:
val sqDistFunc = udf { (v1: Vector, v2: Vector) => Vectors.sqdist(v1, v2) }

val resultDf = leftDf.top_k_join(
  k = lit(-1),
  right = rightDf,
  joinExpr = leftDf("group") === rightDf("group"),
  score = sqDistFunc(leftDf("vector"), rightDf("vector")).as("score")
)

scala> resultDf.show
+----+-------------------+------+-----+-----------------+-----+--------+-----------------+
|rank|              score|userId|group|           vector|group|position|           vector|
+----+-------------------+------+-----+-----------------+-----+--------+-----------------+
|   1|0.13999999999999996|     5|    c|[0.4,0.5,0.6,0.2]|    c|   pos-9|[0.7,0.5,0.8,0.3]|
|   1|0.39999999999999997|     6|    c|[0.3,0.9,1.0,0.1]|    c|   pos-9|[0.7,0.5,0.8,0.3]|
|   1|0.42000000000000004|     2|    b|[0.2,0.3,0.4,0.1]|    b|   pos-7|[0.1,0.8,0.4,0.5]|
|   1|0.15000000000000002|     1|    a|[1.0,0.5,0.6,0.2]|    a|   pos-2|[0.9,0.2,0.8,0.3]|
|   1|               0.27|     3|    a|[0.8,0.4,0.2,0.6]|    a|   pos-1|[0.3,0.4,0.3,0.5]|
|   1|0.04000000000000003|     4|    a|[0.2,0.7,0.4,0.8]|    a|   pos-4|[0.1,0.8,0.5,0.7]|
+----+-------------------+------+-----+-----------------+-----+--------+-----------------+
```

