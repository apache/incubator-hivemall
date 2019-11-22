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

## training

For multiclass classification, the following two objects are supported in xgboost:

- `multi:softmax` set XGBoost to do multiclass classification using the softmax objective, you also need to set `num_class` (number of classes).
- `multi:softprob` same as softmax, but output a vector of `ndata * nclass`, which can be further reshaped to `ndata * nclass` matrix. The result contains predicted probability of each data point belonging to each class.

```sql
select count(distinct label) from news20mc_train;
> 20

-- explicitly use 3 reducers
-- set mapred.reduce.tasks=3;

drop table xgb_softmax_model;
create table xgb_softmax_model as
select 
  train_xgboost(features, label, '-objective multi:softmax -num_class 20 -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select features, (label - 1) as label
  from news20mc_train
  cluster by rand(43) -- shuffle data to reducers
) shuffled;

drop table xgb_softprob_model;
create table xgb_softprob_model as
select 
  train_xgboost(features, label, '-objective multi:softprob -num_class 20 -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select features, (label - 1) as label
  from news20mc_train
  cluster by rand(43) -- shuffle data to reducers
) shuffled;
```

> #### Caution
> `-num_class` is required for multiclass objectives.
> The target label must be in range `[0, num_class)` (i.e., 0, 1, 2, .., num_class-1).

## prediction

```sql
drop table xgb_softmax_predicted;
create table xgb_softmax_predicted as
select
  rowid,
  majority_vote(cast(predicted as int) + 1) as label
from (
  select
    xgboost_predict_one(rowid, features, model_id, model) as (rowid, predicted)
  from
    xgb_softmax_model l
    LEFT OUTER JOIN news20mc_test r
) t
group by rowid;


drop table xgb_softprob_predicted;
create table xgb_softprob_predicted as
select
  rowid,
  array_avg(predicted) as prob,
  argmax(array_avg(predicted)) + 1 as label -- convert 0 start index to 1 start index
from (
  select
    xgboost_predict(rowid, features, model_id, model) as (rowid, predicted)
  from
    xgb_softprob_model l
    LEFT OUTER JOIN news20mc_test r
) t
group by
  rowid;
```

> #### Caution
> For `-objective softmax`, xgboost predictor returns class label in double.
> For `-objective softprob`, probabilities for each label is returned in `array<double>`.

## evaluation

```sql
WITH validate as (
  select 
    t.label as actual, 
    p.label as predicted
  from 
    news20mc_test t
    JOIN xgb_softmax_predicted p
      on (t.rowid = p.rowid)
)
select 
  sum(if(actual=predicted,1.0,0.0))/count(1) 
from
  validate;

WITH validate as (
  select 
    t.label as actual, 
    p.label as predicted
  from 
    news20mc_test t
    JOIN xgb_softprob_predicted p
      on (t.rowid = p.rowid)
)
select 
  sum(if(actual=predicted,1.0,0.0))/count(1) 
from
  validate;
```

|objective| accuracy|
|:-:|:-:|
| softmax | 0.6689206110693713999 |
| softprob | 0.6944653143000250438 |