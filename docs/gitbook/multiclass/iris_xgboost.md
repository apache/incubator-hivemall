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

## Data preparation

```sql
create database iris;
use iris;

create external table raw (
  sepal_length int,
  sepal_width int,
  petal_length int,
  petak_width int,
  class string
)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '/dataset/iris/raw';

$ sed '/^$/d' iris.data | hadoop fs -put - /dataset/iris/raw/iris.data
```

```sql
drop table label_mapping;
create table label_mapping 
as
select
  class,
  rank - 1 as label -- zero start index
from (
  select
    distinct class,
    dense_rank() over (order by class) as rank
  from 
    raw
) t;
```

```sql
drop table xgb_input;
create table xgb_input
as
select
  rowid() as rowid,
  array(sepal_length, sepal_width, petal_length, petal_width) as dense_features,
  indexed_features(sepal_length, sepal_width, petal_length, petal_width) as sparse_features,
  t2.label
from
  raw t1
  JOIN label_mapping t2 ON (t1.class = t2.class)
;

select * from xgb_input limit 3;
```

| xgb_input.rowid | xgb_input.dense_features  |      xgb_input.sparse_features  |    xgb_input.label |
|:-:|:-:|:-:|:-:|
| 1-1 |   [5,3,1,0] |     ["1:5","2:3","3:1","4:0"]   |   0 |
| 1-2 |   [4,3,1,0] |     ["1:4","2:3","3:1","4:0"]   |   0 |
| 1-3 |   [4,3,1,0] |     ["1:4","2:3","3:1","4:0"]   |   0 |

## Training

```
-- explicitly use 3 reducers
-- set mapred.reduce.tasks=3;

drop table xgb_softmax_model;
create table xgb_softmax_model 
as
select 
  train_xgboost(features, label, '-objective multi:softmax -num_class 3 -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select 
    -- both sparse and dense format is supported
    dense_features as features, label
    -- sparse_features as features, label 
  from
    xgb_input
  cluster by rand(43) -- shuffle
) shuffled;
```

> #### Caution
> `-num_class` is required for multiclass objectives.
> Note both sparse and dense vector is supported for feature vector format.

## Prediction

```sql
drop table xgb_softmax_predicted;
create table xgb_softmax_predicted as
select
  rowid,
  majority_vote(cast(predicted as int)) as label
from (
  select
    xgboost_predict_one(rowid, dense_features, model_id, model) as (rowid, predicted)
    -- xgboost_predict_one(rowid, sparse_features, model_id, model) as (rowid, predicted)
  from
    xgb_softmax_model l
    LEFT OUTER JOIN xgb_input r
) t
group by rowid;
```

## Evaluation

```sql
WITH validate as (
  select 
    t.label as actual, 
    p.label as predicted
  from 
    xgb_input t
    JOIN xgb_softmax_predicted p
      on (t.rowid = p.rowid)
)
select 
  sum(if(actual=predicted,1.0,0.0))/count(1) 
from
  validate;
```

> 0.9533333333333333333