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

This page shows the usage of General Binary Classifier using a9a dataset.

<!-- toc -->

# Training

```sql
create table model
as
select 
 feature,
 avg(weight) as weight
from (
  select 
     train_classifier(
       add_bias(features), label, 
       "-loss logistic -iter 30"
     ) as (feature,weight)
  from 
     a9a_train
 ) t 
group by feature;
```

# Prediction

```sql
create table predict 
as
WITH exploded as (
select 
  rowid,
  label,
  extract_feature(feature) as feature,
  extract_weight(feature) as value
from 
  a9a_test LATERAL VIEW explode(add_bias(features)) t AS feature
)
select
  t.rowid, 
  sigmoid(sum(m.weight * t.value)) as prob,
  (case when sigmoid(sum(m.weight * t.value)) >= 0.5 then 1.0 else 0.0 end) as label
from 
  exploded t LEFT OUTER JOIN
  model m ON (t.feature = m.feature)
group by
  t.rowid;
```

# Evaluation

```sql
create or replace view submit as
select 
  t.label as actual, 
  p.label as predicted, 
  p.prob as probability
from 
  a9a_test t 
  JOIN predict p on (t.rowid = p.rowid);

select 
  sum(if(actual == predicted, 1, 0)) / count(1) as accuracy
from
  submit;
```

> 0.8462625145875561

The following table shows accuracy for changing optimizer by `-loss logistic -opt XXXXXX -reg l1 -iter 30` option:

| Optimizer | Accuracy |
|:--:|:--:|
| Default (Adagrad+RDA) | 0.8462625145875561 |
| SGD | 0.8462010932989374 |
| Momentum | 0.8254406977458387 |
| Nesterov | 0.8286346047540077 |
| AdaGrad | 0.850991953811191 |
| RMSprop | 0.8463239358761747 |
| RMSpropGraves | 0.825563540323076 |
| AdaDelta | 0.8492721577298692 |
| Adam | 0.8341625207296849 |
| Nadam | 0.8349609974817271 |
| Eve | 0.8348381549044899 |
| AdamHD | 0.8447269823720902 |

> #### Note
> Optimizers using momentum need to tune decay rate well.
> Default (Adagrad+RDA), AdaDelta, Adam, and AdamHD is worth trying in my experience.

