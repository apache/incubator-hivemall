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

<!-- toc -->

# Notice

* `top_k_join` is supported in the DataFrame of Spark v2.1.0 or later.
* A type of `score` must be ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, or DecimalType.
* If `k` is less than 0, the order is reverse and `top_k_join` joins the tail-K records of `rightDf`.

# Usage

For example, we have two tables below;

- An input table (`leftDf`)

| userId | group | x   | y   |
|:------:|:-----:|:---:|:---:|
| 1      | b     | 0.3 | 0.3 |
| 2      | a     | 0.5 | 0.4 |
| 3      | a     | 0.1 | 0.8 |
| 4      | c     | 0.2 | 0.2 |
| 5      | a     | 0.1 | 0.4 |
| 6      | b     | 0.8 | 0.3 |

- A reference table (`rightDf`)

| group | position | x   | y   |
|:-----:|:--------:|:---:|:---:|
| a     | pos-1    | 0.0 | 0.1 |
| a     | pos-2    | 0.9 | 0.3 |
| a     | pos-3    | 0.3 | 0.2 |
| b     | pos-4    | 0.5 | 0.7 |
| b     | pos-5    | 0.4 | 0.2 |
| c     | pos-6    | 0.8 | 0.7 |
| c     | pos-7    | 0.3 | 0.3 |
| c     | pos-8    | 0.4 | 0.2 |
| c     | pos-9    | 0.3 | 0.8 |

In the two tables, the example computes the nearest `position` for `userId` in each `group`.
The standard way using DataFrame window functions would be as follows:

```
val computeDistanceFunc =
  sqrt(pow(inputDf("x") - masterDf("x"), lit(2.0)) + pow(inputDf("y") - masterDf("y"), lit(2.0)))

leftDf.join(
    right = rightDf,
    joinExpr = leftDf("group") === rightDf("group")
  )
  .select(inputDf("group"), $"userId", $"posId", computeDistanceFunc.as("score"))
  .withColumn("rank", rank().over(Window.partitionBy($"group", $"userId").orderBy($"score".desc)))
  .where($"rank" <= 1)
```

You can use `top_k_join` as follows:

```
leftDf.top_k_join(
    k = lit(-1),
    right = rightDf,
    joinExpr = leftDf("group") === rightDf("group"),
    score = computeDistanceFunc.as("score")
  )
```

The result is as follows:

| rank | score | userId | group | x   | y   | group | position | x   | y   |
|:----:|:-----:|:------:|:-----:|:---:|:---:|:-----:|:--------:|:---:|:---:|
| 1    | 0.100 | 4      | c     | 0.2 | 0.2 | c     | pos9     | 0.3 | 0.8 |
| 1    | 0.100 | 1      | b     | 0.3 | 0.3 | b     | pos5     | 0.4 | 0.2 |
| 1    | 0.300 | 6      | b     | 0.8 | 0.8 | b     | pos4     | 0.5 | 0.7 |
| 1    | 0.200 | 2      | a     | 0.5 | 0.4 | a     | pos3     | 0.3 | 0.2 |
| 1    | 0.100 | 3      | a     | 0.1 | 0.8 | a     | pos1     | 0.0 | 0.1 |
| 1    | 0.100 | 5      | a     | 0.1 | 0.4 | a     | pos1     | 0.0 | 0.1 |

