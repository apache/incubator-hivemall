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

Hivemall Random Forest supports libsvm-like sparse inputs. This page shows a classification example on 20-newsgroup dataset.

> #### Note
> This feature, i.e., Sparse input support in Random Forest, is supported since Hivemall v0.5.0 or later.
> [`feature_hashing`](http://hivemall.incubator.apache.org/userguide/ft_engineering/hashing.html#featurehashing-function) function is useful to prepare feature vectors for Random Forest.

<!-- toc -->

## Training

```sql
drop table rf_model;
create table rf_model
as
select
  train_randomforest_classifier(
    features,
    convert_label(label),  -- convert -1/1 to 0/1
    '-trees 50 -seed 71'   -- hyperparameters
  )
from
  train;
```

> #### Caution
> label must be in `[0, k)` where `k` is the number of classes.

## Prediction

```sql
-- SET hivevar:classification=true;

drop table rf_predicted;
create table rf_predicted
as
SELECT
  rowid,
  rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  -- rf_ensemble(predicted.value, predicted.posteriori) as predicted -- avoid OOB accuracy (i.e., model_weight)
FROM (
  SELECT
    rowid, 
    m.model_weight,
	-- v0.5.0 and later
    tree_predict(m.model_id, m.model, t.features, "-classification") as predicted
    -- before v0.5.0
	-- tree_predict(m.model_id, m.model, t.features, ${classification}) as predicted
  FROM
    rf_model m
    LEFT OUTER JOIN -- CROSS JOIN
    test t
) t1
group by
  rowid
;
```

## Evaluation

```sql
WITH submit as (
  select 
    convert_label(t.label) as actual, 
    p.predicted.label as predicted
  from 
    test t 
    JOIN rf_predicted p on (t.rowid = p.rowid)
)
select
  sum(if(actual = predicted, 1, 0)) / count(1) as accuracy
from
  submit;
```

> 0.8112489991993594
