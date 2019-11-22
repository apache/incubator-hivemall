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

This tutorial explains how to use XGBoost for regression problems.

<!-- toc -->

## Training

The following objective is supported in XGboost for regression:

- `reg:squarederror` regression with squared loss
- `reg:logistic` logistic regression

`reg:squarederror` is widely used as the regression objective.

```sql
use e2006;

desc e2006tfidf_train;
```

| col_name  | data_type  |
|:-:|:-:|
| rowid |                  int              |                        
| target |                 float            |                      
| features |               array<string>    |


```sql
-- explicitly use 3 reducers
-- set mapred.reduce.tasks=3

drop table xgb_regr_model;
create table xgb_regr_model as
select 
  train_xgboost(features, target, '-objective reg:squarederror -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select features, target
  from e2006tfidf_train
  cluster by rand(43) -- shuffle data to reducers
) shuffled;
```

## prediction

```sql
drop table xgb_regr_predicted;
create table xgb_regr_predicted as
select
  rowid,
  avg(predicted) as predicted
from (
  select
    xgboost_predict_one(rowid, features, model_id, model) as (rowid, predicted)
  from
    xgb_regr_model l
    LEFT OUTER JOIN e2006tfidf_test r
) t
group by rowid;
```

> #### Note
> `xgboost_predict` returns new double[1] (e.g., [-3.9760303385555744]) for `-objective reg:squarederror`.
> On the other hand, `xgboost_predict_one` returns a scalar double value as predicted `-3.9760303385555744`.

## evaluation

```sql
WITH submit as (
  select 
    t.target as actual, 
    p.predicted as predicted
  from 
    e2006tfidf_test t
    JOIN xgb_regr_predicted p 
      on (t.rowid = p.rowid)
)
select 
   rmse(predicted, actual) as RMSE,
   mse(predicted, actual) as MSE, 
   mae(predicted, actual) as MAE,
   r2(predicted, actual) as R2
from 
   submit;
```

| rmse | mse | mae | r2 |
|:-:|:-:|:-:|:-:|
| 0.3949633797136429 | 0.15599607131482326 | 0.25367043577533693 | 0.4603881976325721 |
