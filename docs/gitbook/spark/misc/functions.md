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

flatten
================

`df.flatten()` flattens a nested schema of `df` into a flat one.

## Usage

```scala
scala> val df = Seq((0, (1, (3.0, "a")), (5, 0.9))).toDF()
scala> df.printSchema
root
 |-- _1: integer (nullable = false)
 |-- _2: struct (nullable = true)
 |    |-- _1: integer (nullable = false)
 |    |-- _2: struct (nullable = true)
 |    |    |-- _1: double (nullable = false)
 |    |    |-- _2: string (nullable = true)
 |-- _3: struct (nullable = true)
 |    |-- _1: integer (nullable = false)
 |    |-- _2: double (nullable = false)

scala> df.flatten(separator = "$").printSchema
root
 |-- _1: integer (nullable = false)
 |-- _2$_1: integer (nullable = true)
 |-- _2$_2$_1: double (nullable = true)
 |-- _2$_2$_2: string (nullable = true)
 |-- _3$_1: integer (nullable = true)
 |-- _3$_2: double (nullable = true)
```

from_csv
================

This function parses a column containing a CSV string into a `StructType`
with the specified schema.

## Usage

```scala
scala> val df = Seq("1, abc, 0.8").toDF()

scala> df.printSchema
root
 |-- value: string (nullable = true)

scala> val schema = new StructType().add("a", IntegerType).add("b", StringType).add("c", DoubleType)

scala> df.select(from_csv($"value", schema)).printSchema
root
 |-- csvtostruct(value): struct (nullable = true)
 |    |-- a: integer (nullable = true)
 |    |-- b: string (nullable = true)
 |    |-- c: double (nullable = true)

scala> df.select(from_csv($"value", schema)).show
+------------------+
|csvtostruct(value)|
+------------------+
|      [1, abc,0.8]|
+------------------+
```

to_csv
================

This function converts a column containing a `StructType` into a CSV string
with the specified schema.

## Usage

```scala
scala> val df = Seq((1, "a", (0, 3.9, "abc")), (8, "c", (2, 0.4, "def"))).toDF()

scala> df.printSchema
root
 |-- _1: integer (nullable = false)
 |-- _2: string (nullable = true)
 |-- _3: struct (nullable = true)
 |    |-- _1: integer (nullable = false)
 |    |-- _2: double (nullable = false)
 |    |-- _3: string (nullable = true)

scala> df.select(to_csv($"_3"))

scala> df.select(to_csv($"_3")).printSchema
root
 |-- structtocsv(_3): string (nullable = true)

scala> df.select(to_csv($"_3")).show
+---------------+
|structtocsv(_3)|
+---------------+
|      0,3.9,abc|
|      2,0.4,def|
+---------------+
```
