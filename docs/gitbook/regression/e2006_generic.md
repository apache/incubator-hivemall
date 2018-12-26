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

This tutorial shows how to apply General Regressor for a regression problem of e2006 dataset.

<!-- toc -->

## Training
```sql
set mapred.reduce.tasks=32;

drop table e2006tfidf_generic_model;
create table e2006tfidf_generic_model as
select 
 feature,
 avg(weight) as weight
from 
 (select 
     train_regressor(
       add_bias(features), target,
       '-loss squaredloss -opt AdamHD -reg No -iters 20'
     ) as (feature, weight)
  from 
     e2006tfidf_train_x3
 ) t 
group by feature;

-- reset to the default setting
set mapred.reduce.tasks=-1;
```

> #### Caution
> Regularization could not work well for regression problem. Then, try providing `-reg No` option as seen in the above query.
> Also, do not use `voted_avg()` for regression. `voted_avg()` is for classification.

## prediction
```sql
create or replace view e2006tfidf_generic_predict
as
select
  t.rowid, 
  sum(m.weight * t.value) as predicted
from 
  e2006tfidf_test_exploded t LEFT OUTER JOIN
  e2006tfidf_generic_model m ON (t.feature = m.feature)
group by
  t.rowid;
```

## evaluation
```sql
WITH submit as (
  select 
    t.target as actual, 
    p.predicted as predicted
  from 
    e2006tfidf_test t
    JOIN e2006tfidf_generic_predict p 
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
| 0.37125069279938866 | 0.13782707690402607 | 0.2270351090214029 | 0.5232372408076887 |


