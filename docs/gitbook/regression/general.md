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

In our regression tutorials, you can tackle realistic prediction problems by using several Hivemall's regression features such as:

- [PA1a](e2006_arow.html#pa1a)
- [PA2a](e2006_arow.html#pa2a)
- [AROW](e2006_arow.html#arow)
- [AROWe](e2006_arow.html#arowe)

Our `train_regressor` function enables you to solve the regression problems with flexible configurable options. Let us try the function below.

It should be noted that the sample queries require you to prepare [E2006-tfidf data](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-tfidf). See [our E2006-tfidf tutorial page](../regression/e2006_dataset.md) for further instructions.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

# Training

```sql
create table e2006tfidf_regression_model as
select 
	feature,
	avg(weight) as weight
from (
	select 
  	train_regressor(features,target,'-loss squaredloss -opt AdaGrad -reg no') as (feature,weight)
  from 
    e2006tfidf_train_x3
) t 
group by feature;
```

# Prediction & evaluation

```sql
WITH predict as (
	select
	  t.rowid, 
	  sum(m.weight * t.value) as predicted
	from 
	  e2006tfidf_test_exploded t LEFT OUTER JOIN
	  e2006tfidf_regression_model m ON (t.feature = m.feature)
	group by
	  t.rowid
),
submit as (
	select 
	  t.target as actual, 
	  p.predicted as predicted
	from 
	  e2006tfidf_test t JOIN predict p 
	    on (t.rowid = p.rowid)
)
select rmse(predicted, actual) from submit;
```
