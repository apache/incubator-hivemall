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

Feature binning is a method of dividing quantitative variables into categorical values.

*Note: This feature is supported from Hivemall v0.5-rc.1 or later.*

<!-- toc -->

# Usage

Sample data (*user* table)

``` sql
CREATE TABLE user (
  name string, age int, gender string
);
INSERT INTO user VALUES
  ('Jacob', 20, 'Male'),
  ('Mason', 22, 'Male'),
  ('Sophia', 35, 'Female'),
  ('Ethan', 55, 'Male'),
  ('Emma', 15, 'Female'),
  ('Noah', 46, 'Male'),
  ('Isabella', 20, 'Female');
```

## A. Pack features

``` sql
WITH t AS (
  SELECT
    array_concat(
      categorical_features(
        array("name", "gender"), name, gender),
      quantitative_features(
        array("age"), age)) AS features
  FROM
    user
),
bins AS (
  SELECT map("age", build_bins(age, 3)) AS quantiles_map
  FROM user
)
SELECT feature_binning(features, quantiles_map) AS features FROM t JOIN bins;
```

*Result*

| features: `array<features::string>` |
| :-: |
| ["name#Jacob","gender#Male","age:1"] |
| ["name#Mason","gender#Male","age:1"] |
| ["name#Sophia","gender#Female","age:2"] |
| ["name#Ethan","gender#Male","age:2"] |
| ["name#Emma","gender#Female","age:0"] |
| ["name#Noah","gender#Male","age:2"] |
| ["name#Isabella","gender#Female","age:1"] |


## B. Get only transformed values

``` sql
WITH bins AS (
  SELECT build_bins(age, 3) AS quantiles
  FROM user
)
SELECT feature_binning(age, quantiles) AS bin FROM user JOIN bins;
```

*Result*

| bin: `int` |
| :-: |
| 1 |
| 1 |
| 2 |
| 2 |
| 0 |
| 2 |
| 1 |


# Composition

These implementations are based on `org.apache.hadoop.hive.ql.udf.generic.GenericUDAFPercentileApprox`

## [UDAF] `build_bins(weight, num_of_bins[, auto_shrink])`
### Input
| weight: int&#124;bigint&#124;float&#124;double | num\_of\_bins: `int` | [auto\_shrink: `boolean` = false] |
| :-: | :-: | :-: |
| weight | 2 <= | behavior when separations are repeated: T=\>skip, F=\>exception |

### Output
| quantiles: `array<double>` |
| :-: |
| array of separation value |

### About `auto_shrink`

There is the possibility quantiles are repeated because of too many `num_of_bins` or too few data.
If `auto_shrink` is true, skip duplicated quantiles. If not, throw an exception.


## [UDF] `feature_binning(features, quantiles_map)/(weight, quantiles)`
### Input
#### A
| features: `array<features::string>` | quantiles\_map: `map<string, array<double>>` |
| :-: | :-: |
| serialized feature | entry:: key: col name, val: quantiles |

#### B
| weight: int&#124;bigint&#124;float&#124;double | quantiles: `array<double>` |
| :-: | :-: |
| weight | array of separation value |

### Output
#### A
| features: `array<feature::string>` |
| :-: |
| serialized and binned features |

#### B
| bin: `int` |
| :-: |
| categorical value(bin ID) |
