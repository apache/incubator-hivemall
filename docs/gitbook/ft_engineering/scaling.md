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

<!-- toc -->

# L1/L2 Normalization

[L1](http://mathworld.wolfram.com/L1-Norm.html) and [L2](http://mathworld.wolfram.com/L2-Norm.html) normalization ensures that each feature vector has unit length:

```sql
select l1_normalize(array('apple:1.0', 'banana:0.5'))
```

> ["apple:0.6666667","banana:0.33333334"]

```sql
select l2_normalize(array('apple:1.0', 'banana:0.5'))
```

> ["apple:0.8944272","banana:0.4472136"]

# Min-Max Normalization

[Min-max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling) converts values to range `[0.0,1.0]`.

```sql
select 
  rescale(target, min(target) over (), max(target) over ()) as target
from
  e2006tfidf_train
```

It can also expressed without Windowing function as follows:

```sql
select min(target), max(target)
from (
  select target from e2006tfidf_train 
-- union all
-- select target from e2006tfidf_test 
) t;
```

> -7.899578       -0.51940954

```sql
set hivevar:min_target=-7.899578;
set hivevar:max_target=-0.51940954;

create or replace view e2006tfidf_train_scaled 
as
select 
  rowid,
  rescale(target, ${min_target}, ${max_target}) as target, 
  features
from 
  e2006tfidf_train;
```

# Feature scaling by zscore

Refer [this article](https://en.wikipedia.org/wiki/Standard_score) to get details about Zscore.

```sql
select 
  zscore(target, avg(target) over (), stddev_pop(target) over ()) as target
from 
  e2006tfidf_train;
```

# Apply Normalization to more complex feature vector

Apply normalization to the following data.

```sql
create table train as 
select 
  1 as rowid, array("weight:69.613","specific_heat:129.07","reflectance:52.111") as features
UNION ALL
select 
  2 as rowid, array("weight:70.67","specific_heat:128.161","reflectance:52.446") as features
UNION ALL
select 
  3 as rowid, array("weight:72.303","specific_heat:128.45","reflectance:52.853") as features

select rowid, features from train;
```

```
1       ["weight:69.613","specific_heat:129.07","reflectance:52.111"]
2       ["weight:70.67","specific_heat:128.161","reflectance:52.446"]
3       ["weight:72.303","specific_heat:128.45","reflectance:52.853"]
```

We can create a normalized table as follows:

```sql
create table train_normalized
as
WITH exploded as (
  select 
    rowid, 
    extract_feature(feature) as feature,
    extract_weight(feature) as value
  from 
    train 
    LATERAL VIEW explode(features) exploded AS feature
), 
scaled as (
  select 
    rowid,
    feature,
    rescale(value, min(value) over (partition by feature), max(value) over (partition by feature)) as minmax,
    zscore(value, avg(value) over (partition by feature), stddev_pop(value) over (partition by feature)) as zscore
  from 
    exploded
)
select
  rowid,
  collect_list(feature(feature, minmax)) as features
from
  scaled
group by
  rowid;
```

```
1       ["reflectance:0.0","specific_heat:1.0","weight:0.0"]
2       ["reflectance:0.4514809","specific_heat:0.0","weight:0.39293614"]
3       ["reflectance:1.0","specific_heat:0.31792927","weight:1.0"]
...
```
