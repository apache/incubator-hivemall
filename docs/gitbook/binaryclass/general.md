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

Hivemall has a generic function for classification: `train_classifier`. Compared to the other functions we will see in the later chapters, `train_classifier` provides simpler and configureable generic interface which can be utilized to build binary classification models in a variety of settings.

Here, we briefly introduce usage of the function. Before trying sample queries, you first need to prepare [a9a data](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a). See [our a9a tutorial page](a9a_dataset.md) for further instructions.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

# Preparation

- Set `total_steps` ideally be `count(1) / {# of map tasks}`:
	```
	hive> select count(1) from a9a_train; 
	hive> set hivevar:total_steps=32561;
	```
- Set `n_samples` to compute accuracy of prediction:
	```
	hive> select count(1) from a9a_test;
	hive> set hivevar:n_samples=16281;
	```

# Training

```sql
create table classification_model as
select
 feature,
 avg(weight) as weight
from
 (
  select
    train_classifier(add_bias(features), label, '-loss logloss -opt SGD -reg no -eta simple -total_steps ${total_steps}') as (feature, weight)
  from
     a9a_train
 ) t
group by feature;
```

> #### Note
>
> `-total_steps` option is an optional parameter and training works without it.

# Prediction & evaluation

```sql
WITH test_exploded as (
  select
    rowid,
    label,
    extract_feature(feature) as feature,
    extract_weight(feature) as value
  from
    a9a_test LATERAL VIEW explode(add_bias(features)) t AS feature
),
predict as (
  select
    t.rowid,
    sigmoid(sum(m.weight * t.value)) as prob,
    (case when sigmoid(sum(m.weight * t.value)) >= 0.5 then 1.0 else 0.0 end)as label
  from
    test_exploded t LEFT OUTER JOIN
    classification_model m ON (t.feature = m.feature)
  group by
    t.rowid
),
submit as (
  select
    t.label as actual,
    pd.label as predicted,
    pd.prob as probability
  from
    a9a_test t JOIN predict pd
      on (t.rowid = pd.rowid)
)
select count(1) / ${n_samples} from submit
where actual = predicted;
```

# Comparison with the other binary classifiers

In the next part of this user guide, our binary classification tutorials introduce many different functions:

- [Logistic Regression](a9a_lr.md)
	- and [its mini-batch variant](a9a_minibatch.md)
- [Perceptron](news20_pa.md#perceptron)
- [Passive Aggressive](news20_pa.md#passive-aggressive)
- [CW](news20_scw.md#confidence-weighted-cw)
- [AROW](news20_scw.md#adaptive-regularization-of-weight-vectors-arow)
- [SCW](news20_scw.md#soft-confidence-weighted-scw1)
- [AdaGradRDA](news20_adagrad.md#adagradrda)
- [AdaGrad](news20_adagrad.md#adagrad)
- [AdaDelta](news20_adagrad.md#adadelta)

All of them actually have the same interface, but mathematical formulation and its implementation differ from each other.

In particular, the above sample queries are almost same as [a9a tutorial using Logistic Regression](a9a_lr.md). The difference is only in a choice of training function: `logress()` vs. `train_classifier()`.

However, at the same time, the options `-loss logloss -opt SGD -reg no -eta simple -total_steps ${total_steps}` for `train_classifier` indicates that Hivemall uses the generic classifier as Logistic Regressor (`logress`). Hence, the accuracy of prediction based on either `logress` and `train_classifier` should be same under the configuration.

In addition, `train_classifier` supports the `-mini_batch` option in a similar manner to [what `logress` does](a9a_minibatch.md). Thus, following two training queries show the same results:

```sql
select
	logress(add_bias(features), label, '-total_steps ${total_steps} -mini_batch 10') as (feature, weight)
from
	a9a_train
```

```sql
select
	train_classifier(add_bias(features), label, '-loss logloss -opt SGD -reg no -eta simple -total_steps ${total_steps} -mini_batch 10') as (feature, weight)
from
	a9a_train
```

Likewise, you can generate many different classifiers based on its options.
