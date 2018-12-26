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

In this tutorial, we build a binary classification model using general classifier.

<!-- toc -->

## Training

```sql
-- set mapred.reduce.tasks=3; -- explicitly use 3 reducers

drop table news20b_generic_model;
create table news20b_generic_model as
select 
 feature,
 voted_avg(weight) as weight
from 
 (select 
     train_classifier(
       add_bias(features), label, 
       '-loss logistic -opt AdamHD -reg l1 -iters 20'
     ) as (feature,weight)
  from
     news20b_train_x3
 ) t 
group by feature;
```

> #### Note
> Default (Adagrad+RDA), AdaDelta, Adam, and AdamHD is worth trying in my experience.

## prediction

```sql
create or replace view news20b_generic_predict
as
select
  t.rowid, 
  sum(m.weight * t.value) as total_weight,
  case when sum(m.weight * t.value) > 0.0 then 1 else -1 end as label
from 
  news20b_test_exploded t LEFT OUTER JOIN
  news20b_generic_model m ON (t.feature = m.feature)
group by
  t.rowid;
```

## evaluation

```sql
WITH submit as (
select 
  t.label as actual, 
  p.label as predicted
from 
  news20b_test t 
  JOIN news20b_generic_predict p
    on (t.rowid = p.rowid)
)
select 
  sum(if(actual = predicted, 1, 0)) / count(1) as accuracy
from
  submit;
```

> 0.967173738991193 (`-opt AdamHD -reg l1 `)