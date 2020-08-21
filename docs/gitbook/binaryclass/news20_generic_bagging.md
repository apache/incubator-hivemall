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

Hivemall generally uses model averaging (i.e., model ensemble) for creating a unified prediction model.
In this tutorial, we show how to apply bagging (i.e., prediction ensemble) for making a prediction.

<!-- toc -->

## Training

```sql
-- set mapred.reduce.tasks=3; -- explicitly use 3 reducers

CREATE TABLE bagging_models
as 
WITH train as (
  select 
     train_classifier(
       add_bias(features), label, 
       '-loss logistic -opt AdamHD -reg l1 -iters 20'
     ) as (feature,weight)
  from
     news20b_train_x3
)
select
  taskid() as modelid,
  feature,
  weight
from 
  train;
```

## prediction

```sql
create table bagging_predict
as
WITH weights as (
  select
    t.rowid,
    m.modelid,
    sum(m.weight * t.value) as total_weight
  from
    news20b_test_exploded t 
    LEFT OUTER JOIN
    bagging_models m ON (t.feature = m.feature)
  group by
    rowid, modelid
),
bagging as (
  select
    rowid,
    avg(total_weight) as total_weight
  from 
    weights
  group by
    rowid 
)
select
  rowid,
  max(total_weight) as total_weight, -- max is dummy 
  -- Note: sum(total_weight) > 0.0 equals to sigmoid((total_weight)) > 0.5
  -- https://en.wikipedia.org/wiki/Sigmoid_function
  case when sum(total_weight) > 0.0 then 1 else -1 end as label
from
  bagging
group by
  rowid;
```

## evaluation

```sql
WITH submit as (
  select 
    t.label as actual, 
    p.label as predicted
  from 
    news20b_test t 
    JOIN bagging_predict p on (t.rowid = p.rowid)
)
select 
  sum(if(actual = predicted, 1, 0)) / count(1) as accuracy
from
  submit;
```

> 0.9641713370696557