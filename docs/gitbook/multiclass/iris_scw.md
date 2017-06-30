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

# Training (multiclass classification)

```sql
create table model_scw1 as
select 
 label, 
 feature,
 argmin_kld(weight, covar) as weight
from 
 (select 
     train_multiclass_scw(features, label) as (label, feature, weight, covar)
  from 
     training_x10
 ) t 
group by label, feature;
```

# Predict

```sql
create or replace view predict_scw1
as
select 
  rowid, 
  m.col0 as score, 
  m.col1 as label
from (
select
   rowid, 
   maxrow(score, label) as m
from (
  select
    t.rowid,
    m.label,
    sum(m.weight * t.value) as score
  from 
    test20p_exploded t LEFT OUTER JOIN
    model_scw1 m ON (t.feature = m.feature)
  group by
    t.rowid, m.label
) t1
group by rowid
) t2;
```

# Evaluation

```sql
create or replace view eval_scw1 as
select 
  t.label as actual, 
  p.label as predicted
from 
  test20p t JOIN predict_scw1 p 
    on (t.rowid = p.rowid);

select count(1)/30 from eval_scw1 
where actual = predicted;
```

> 0.9666666666666667
