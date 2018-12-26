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

> #### Caution
>
> `train_adagrad()` became deprecated since v0.5.0 release. Use smarter [general classifier](./a9a_generic.md) instead.

# AdaGradRDA

> #### Note
>
> The current AdaGradRDA implmenetation can only be applied to classification, not to regression, because it uses hinge loss for the loss function.

## model building

```sql
use news20;

drop table news20b_adagrad_rda_model1;
create table news20b_adagrad_rda_model1 as
select 
 feature,
 voted_avg(weight) as weight
from 
 (select 
     train_adagrad_rda(add_bias(features),label) as (feature,weight)
  from 
     news20b_train_x3
 ) t 
group by feature;
```

## prediction

```sql
create or replace view news20b_adagrad_rda_predict1 
as
select
  t.rowid, 
  sum(m.weight * t.value) as total_weight,
  case when sum(m.weight * t.value) > 0.0 then 1 else -1 end as label
from 
  news20b_test_exploded t LEFT OUTER JOIN
  news20b_adagrad_rda_model1 m ON (t.feature = m.feature)
group by
  t.rowid;
```

## evaluation

```sql
create or replace view news20b_adagrad_rda_submit1 as
select 
  t.label as actual, 
  pd.label as predicted
from 
  news20b_test t JOIN news20b_adagrad_rda_predict1 pd 
    on (t.rowid = pd.rowid);
```

```sql
select count(1)/4996 from news20b_adagrad_rda_submit1 
where actual == predicted;
```

> SCW1 0.9661729383506805 

> ADAGRAD+RDA 0.9677742193755005

# AdaGrad

> #### Note
>
> AdaGrad is better suited for a binary classification problem because the current implementation only support logistic loss.

## model building

```sql
drop table news20b_adagrad_model1;
create table news20b_adagrad_model1 as
select 
 feature,
 voted_avg(weight) as weight
from 
 (select 
     train_adagrad_regr(add_bias(features),convert_label(label)) as (feature,weight)
  from 
     news20b_train_x3
 ) t 
group by feature;
```

> #### Caution
> `adagrad` takes 0/1 for a label value and `convert_label(label)` converts a label value from -1/+1 to 0/1.

## prediction

```sql
create or replace view news20b_adagrad_predict1 
as
select
  t.rowid, 
  case when sigmoid(sum(m.weight * t.value)) >= 0.5 then 1 else -1 end as label
from 
  news20b_test_exploded t LEFT OUTER JOIN
  news20b_adagrad_model1 m ON (t.feature = m.feature)
group by
  t.rowid;
```

## evaluation

```sql
create or replace view news20b_adagrad_submit1 as
select 
  t.label as actual, 
  p.label as predicted
from 
  news20b_test t JOIN news20b_adagrad_predict1 p
    on (t.rowid = p.rowid);
```

```sql
select count(1)/4996 from news20b_adagrad_submit1 
where actual == predicted;
```

> 0.9549639711769415 (adagrad)

# AdaDelta

> #### Caution
> AdaDelta can only be applied for regression problem because the current implementation only support logistic loss.

## model building

```sql
drop table news20b_adadelta_model1;
create table news20b_adadelta_model1 as
select 
 feature,
 voted_avg(weight) as weight
from 
 (select 
     adadelta(add_bias(features),convert_label(label)) as (feature,weight)
  from 
     news20b_train_x3
 ) t 
group by feature;
```

## prediction

```sql
create or replace view news20b_adadelta_predict1 
as
select
  t.rowid, 
  case when sigmoid(sum(m.weight * t.value)) >= 0.5 then 1 else -1 end as label
from 
  news20b_test_exploded t LEFT OUTER JOIN
  news20b_adadelta_model1 m ON (t.feature = m.feature)
group by
  t.rowid;
```

## evaluation

```sql
create or replace view news20b_adadelta_submit1 as
select 
  t.label as actual, 
  p.label as predicted
from 
  news20b_test t JOIN news20b_adadelta_predict1 p
    on (t.rowid = p.rowid);
```


```sql
select count(1)/4996 from news20b_adadelta_submit1 
where actual == predicted;
```

_AdaDelta often performs better than AdaGrad._

> 0.9549639711769415 (adagrad)

> 0.9545636509207366 (adadelta)
