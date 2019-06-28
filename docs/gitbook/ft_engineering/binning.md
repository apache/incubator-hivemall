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

Feature binning is a method of dividing quantitative variables into categorical values. It groups quantitative values into a pre-defined number of bins.

If the number of bins is set to 3, the bin ranges become something like `[-Inf, 1], (1, 10], (10, Inf]`.

<!-- toc -->

# Data Preparation

Prepare sample data (*users* table) first as follows:

``` sql
CREATE TABLE users (
  rowid int, name string, age int, gender string
);
INSERT INTO users VALUES
  (1, 'Jacob', 20, 'Male'),
  (2, 'Mason', 22, 'Male'),
  (3, 'Sophia', 35, 'Female'),
  (4, 'Ethan', 55, 'Male'),
  (5, 'Emma', 15, 'Female'),
  (6, 'Noah', 46, 'Male'),
  (7, 'Isabella', 20, 'Female')
;

CREATE TABLE input as
SELECT
  rowid,
  array_concat(
    categorical_features(
      array('name', 'gender'),
      name, gender
    ),
    quantitative_features(
      array('age'),
      age
    )
  ) AS features
FROM
  users;
  
select * from input limit 2;
```

| input.rowid | input.features |
|:--|:--|
|1 | ["name#Jacob","gender#Male","age:20.0"] |
|2 | ["name#Mason","gender#Male","age:22.0"] |

# Usage

## Custom rule for binning

You can provide a custom rule for binning as follows:

```sql
select 
  features as original,
  feature_binning(
    features,
    -- [-INF-10.0], (10.0-20.0], (20.0-30.0], (30.0-40.0], (40.0-INF]
    map('age', array(-infinity(), 10.0, 20.0, 30.0, 40.0, infinity()))
  ) as binned
from
  input;
```

| original | binned |
|:--|:--|
| ["name#Jacob","gender#Male","age:20.0"] | ["name#Jacob","gender#Male","age:1"] |
| ["name#Mason","gender#Male","age:22.0"] | ["name#Mason","gender#Male","age:2"] |
| ["name#Sophia","gender#Female","age:35.0"] | ["name#Sophia","gender#Female","age:3"] |
| ["name#Ethan","gender#Male","age:55.0"] | ["name#Ethan","gender#Male","age:4"] |
| ["name#Emma","gender#Female","age:15.0"] | ["name#Emma","gender#Female","age:1"] |
| ["name#Noah","gender#Male","age:46.0"] | ["name#Noah","gender#Male","age:4"] |
| ["name#Isabella","gender#Female","age:20.0"] | ["name#Isabella","gender#Female","age:1"] |

## Binning based on Quantiles

You can apply feature binning based on [quantiles](https://en.wikipedia.org/wiki/Quantile). 

Suppose converting `age` values into 3 bins:

```sql
SELECT
  map('age', build_bins(age, 3)) AS quantiles_map
FROM
  users
```

> {"age":[-Infinity,18.333333333333332,30.666666666666657,Infinity]}

In the above query result, you can find 4 values for age in `quantiles_map`. It's a threshold for 3 bins.

```sql
WITH bins as (
  SELECT
    map('age', build_bins(age, 3)) AS quantiles_map
  FROM
    users
)
select
  feature_binning(
    array('age:-Infinity', 'age:-1', 'age:0', 'age:1', 'age:18.333333333333331', 'age:18.333333333333332'), quantiles_map
  ),
  feature_binning(
    array('age:18.3333333333333333', 'age:18.33333333333334', 'age:19', 'age:30', 'age:30.666666666666656', 'age:30.666666666666657'), quantiles_map
  ),
  feature_binning(
    array('age:666666666666658', 'age:30.66666666666666', 'age:31', 'age:99', 'age:Infinity'), quantiles_map
  ),
  feature_binning(
    array('age:NaN'), quantiles_map
  ),
  feature_binning( -- not in map
    array('weight:60.3'), quantiles_map
  )
from
  bins
```

> ["age:0","age:0","age:0","age:0","age:0","age:0"]       ["age:0","age:1","age:1","age:1","age:1","age:1"]       ["age:2","a
ge:2","age:2","age:2","age:2"]  ["age:3"]       ["weight:60.3"]

The following query shows more practical usage:

``` sql
WITH bins AS (
  SELECT
    map('age', build_bins(age, 3)) AS quantiles_map
  FROM
    users
)
SELECT
  feature_binning(features, quantiles_map) AS features
FROM
  input
  CROSS JOIN bins;
```

| features: `array<features::string>` |
| :-- |
| ["name#Jacob","gender#Male","age:1"] |
| ["name#Mason","gender#Male","age:1"] |
| ["name#Sophia","gender#Female","age:2"] |
| ["name#Ethan","gender#Male","age:2"] |
| ... |

## Concrete Example

Here, we show a more practical usage of `feature_binning` UDF that applied feature binning for given feature vectors.

```sql
WITH extracted as (
  select 
    extract_feature(feature) as index,
    extract_weight(feature) as value
  from
    input l
    LATERAL VIEW explode(features) r as feature
  where
    instr(feature, ':') > 0 -- filter out categorical features
),
mapping as (
  select
    index, 
    build_bins(value, 5, true) as quantiles -- 5 bins with auto bin shrinking
  from
    extracted
  group by
    index
),
bins as (
   select 
    to_map(index, quantiles) as quantiles 
   from
    mapping
)
select
  l.features as original,
  feature_binning(l.features, r.quantiles) as features
from
  input l
  cross join bins r
-- limit 10;
```

| original | features |
|:--|:--|
| ["name#Jacob","gender#Male","age:20.0"] | ["name#Jacob","gender#Male","age:2"] |
| ["name#Isabella","gender#Female","age:20.0"] | ["name#Isabella","gender#Female","age:2"] |
| ... | ... |


## Create a mapping table by Feature Binning

```sql
WITH bins AS (
  SELECT build_bins(age, 3) AS quantiles
  FROM users
)
SELECT
  age, feature_binning(age, quantiles) AS bin
FROM
  users CROSS JOIN bins;
```

| age:` int` | bin: `int` |
|:-:|:-:|
| 20 | 1 |
| 22 | 1 |
| 35 | 2 |
| 55 | 2 |
| 15 | 0 |
| 46 | 2 |
| 20 | 1 |

# Function Signatures

### UDAF `build_bins(weight num_of_bins [, auto_shrink=false])`

#### Input

| weight: int&#124;bigint&#124;float&#124;double | num\_of\_bins: `int` | [auto\_shrink: `boolean` = false] |
| :-: | :-: | :-: |
| weight | greather than or equals to 2 | behavior when separations are repeated: T=\>skip, F=\>exception |

#### Output

| quantiles: `array<double>` |
| :-: |
| thresholds of bins based on quantiles |

> #### Note
> There is the possibility quantiles are repeated because of too many `num_of_bins` or too few data.
> If `auto_shrink` is set to true, skip duplicated quantiles. If not, throw an exception.

### UDF `feature_binning(features, quantiles_map)`

#### Input 

| features: `array<features::string>` | quantiles\_map: `map<string, array<double>>` |
| :-: | :-: |
| feature vector | a map where key=column name and value=quantiles |

#### Output

| features: `array<feature::string>` |
| :-: |
| binned features |

### UDF `feature_binning(weight, quantiles)`

#### Input

| weight: int&#124;bigint&#124;float&#124;double | quantiles: `array<double>` |
| :-: | :-: |
| weight | array of separation value |

#### Output

| bin: `int` |
| :-: |
| categorical value (bin ID) |
