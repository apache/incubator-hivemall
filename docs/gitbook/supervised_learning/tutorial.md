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

# Step-by-Step Tutorial on Supervised Learning

<!-- toc -->

## What is Hivemall?

[Apache Hivemall](https://github.com/apache/incubator-hivemall) is a collection of user-defined functions (UDFs) for HiveQL which is strongly optimized for machine learning (ML) and data science. To give an example, you can efficiently build a logistic regression model with the stochastic gradient descent (SGD) optimization by issuing the following ~10 lines of query:

```sql
SELECT
  train_classifier(
    features,
    label,
    '-loss_function logloss -optimizer SGD'
  ) as (feature, weight)
FROM
  training
;
```

Below we list ML and relevant problems that Hivemall can solve:

- [Binary and multi-class classification](../binaryclass/general.html)
- [Regression](../regression/general.html)
- [Recommendation](../recommend/cf.html)
- [Anomaly detection](../anomaly/lof.html)
- [Natural language processing](../misc/tokenizer.html)
- [Clustering](../misc/tokenizer.html) (i.e., topic modeling)
- [Data sketching](../misc/funcs.html#sketching)
- Evaluation

Our [YouTube demo video](https://www.youtube.com/watch?v=cMUsuA9KZ_c) would be helpful to understand more about an overview of Hivemall.

This tutorial explains the basic usage of Hivemall with examples of supervised learning of simple regressor and binary classifier.

## Binary classification

Imagine a scenario that we like to build a binary classifier from the mock `purchase_history` data and predict unforeseen purchases to conduct a new campaign effectively:

| day\_of\_week | gender | price | category | label |
|:---:|:---:|:---:|:---:|:---|
|Saturday | male | 600 | book | 1 |
|Friday | female | 4800 | sports | 0 |
|Friday | other | 18000  | entertainment | 0 |
|Thursday | male | 200 | food | 0 |
|Wednesday | female | 1000 | electronics | 1 |

You can create this table as follows:

```sql
create table if not exists purchase_history as
select 1 as id, "Saturday" as day_of_week, "male" as gender, 600 as price, "book" as category, 1 as label
union all
select 2 as id, "Friday" as day_of_week, "female" as gender, 4800 as price, "sports" as category, 0 as label
union all
select 3 as id, "Friday" as day_of_week, "other" as gender, 18000 as price, "entertainment" as category, 0 as label
union all
select 4 as id, "Thursday" as day_of_week, "male" as gender, 200 as price, "food" as category, 0 as label
union all
select 5 as id, "Wednesday" as day_of_week, "female" as gender, 1000 as price, "electronics" as category, 1 as label
;
```

Use Hivemall [`train_classifier()`](../misc/funcs.html#binary-classification) UDF to tackle the problem as follows.

### Step 1. Feature representation

First of all, we have to convert the records into pairs of the feature vector and corresponding target value. Here, Hivemall requires you to represent input features in a specific format.

To be more precise, Hivemall represents single feature in a concatenation of **index** (i.e., **name**) and its **value**:

- Quantitative feature: `<index>:<value>`
  - e.g., `price:600.0`
- Categorical feature: `<index>#<value>`
  - e.g., `gender#male`

Feature index and feature value are separated by comma. When comma is omitted, the value is considered to be `1.0`. So, a categorical feature `gender#male` a [one-hot representation](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science) of `index := gender#male` and `value := 1.0`. Note that `#` is not a special character for categorical feature.

Each of those features is a string value in Hive, and "feature vector" means an array of string values like:

```
["price:600.0", "day of week#Saturday", "gender#male", "category#book"]
```

See also more detailed [document for input format](../getting_started/input-format.html).

Therefore, what we first need to do is to convert the records into an array of feature strings, and Hivemall functions [`quantitative_features()`](../getting_started/input-format.html#quantitative-features), [`categorical_features()`](../getting_started/input-format.html#categorical-features) and [`array_concat()`](../misc/generic_funcs.html#array) provide a simple way to create the pairs of feature vector and target value:

```sql
create table if not exists training as
select
  id,
  array_concat( -- concatenate two arrays of quantitative and categorical features into single array
    quantitative_features(
      array("price"), -- quantitative feature names
      price -- corresponding column names
    ),
    categorical_features(
      array("day of week", "gender", "category"), -- categorical feature names
      day_of_week, gender, category -- corresponding column names
    )
  ) as features,
  label
from
  purchase_history
;
```

The training table is as follows:

|id | features |  label |
|:---:|:---|:---|
|1 |["price:600.0","day of week#Saturday","gender#male","category#book"] | 1 |
|2 |["price:4800.0","day of week#Friday","gender#female","category#sports"] |  0 |
|3 |["price:18000.0","day of week#Friday","gender#other","category#entertainment"]| 0 |
|4 |["price:200.0","day of week#Thursday","gender#male","category#food"] | 0 |
|5 |["price:1000.0","day of week#Wednesday","gender#female","category#electronics"]| 1 |

The output table `training` will be directly used as an input to Hivemall's ML functions in the next step.

> #### Note
>
> You can apply extra Hivemall functions (e.g., [`rescale()`](../misc/funcs.html#feature-scaling), [`feature_hashing()`](../misc/funcs.html#feature-hashing), [`l1_normalize()`](../misc/funcs.html#feature-scaling)) for the features in this step to make your prediction model more accurate and stable; it is known as *feature engineering* in the context of ML. See our [documentation](../ft_engineering/scaling.html) for more information.

### Step 2. Training

Once the original table `purchase_history` has been converted into pairs of `features` and `label`, you can build a binary classifier by running the following query:

```sql
create table if not exists classifier as
select
  train_classifier(
    features, -- feature vector
    label, -- target value
    '-loss_function logloss -optimizer SGD -regularization l1' -- hyper-parameters
  ) as (feature, weight)
from
  training
;
```

What the above query does is to build a binary classifier with:

- `-loss_function logloss`
  - Use logistic loss i.e., logistic regression
- `-optimizer SGD`
  - Learn model parameters with the SGD optimization
- `-regularization l1`
  - Apply L1 regularization

Eventually, the output table `classifier` stores model parameters as:

| feature | weight |
|:---:|:---:|
| day of week#Wednesday | 0.7443372011184692 |
| day of week#Thursday | 1.415687620465178e-07 |
| day of week#Friday | -0.2697019577026367 |
| day of week#Saturday | 0.7337419390678406 |
| category#book | 0.7337419390678406 |
| category#electronics | 0.7443372011184692 |
| category#entertainment | 5.039264578954317e-07 |
| category#food | 1.415687620465178e-07 |
| category#sports | -0.2697771489620209 |
| gender#male | 0.7336684465408325 |
| gender#female | 0.47442761063575745 |
| gender#other | 5.039264578954317e-07 |
| price | -110.62307739257812 |

Notice that weight is learned for each possible value in a categorical feature, and for every single quantitative feature.

Of course, you can optimize hyper-parameters to build more accurate prediction model. Check the output of the following query to see all available options, including learning rate, number of iterations and regularization parameters, and their default values:

```sql
select train_classifier('-help');
-- Hivemall 0.5.2 and before
-- select train_classifier(array(), 0, '-help');
```

### Step 3. Prediction

Now, the table `classifier` has liner coefficients for given features, and we can predict unforeseen samples by computing a weighted sum of their features.

How about the probability of purchase by a `male` customer who sees a `food` product priced at `120` on `Friday`? Which product is more likely to be purchased by the customer on `Friday`?

To differentiate potential purchases, create a `unforeseen_samples` table with these unknown combinations of features:

```sql
create table if not exists unforeseen_samples as
select 1 as id, array("gender#male", "category#food", "day of week#Friday", "price:120") as features
union all
select 2 as id, array("gender#male", "category#sports", "day of week#Friday", "price:1000") as features
union all
select 3 as id, array("gender#male", "category#electronics", "day of week#Friday", "price:540") as features
;
```

Prediction for the feature vectors can be made by join operation between `unforeseen_samples` and `classifier` on each feature as:

```sql
with features_exploded as (
  select
    id,
    -- split feature string into its name and value
    -- to join with a model table
    extract_feature(fv) as feature,
    extract_weight(fv) as value
  from
    unforeseen_samples t1
    LATERAL VIEW explode(features) t2 as fv
)
select
  t1.id,
  sigmoid( sum(p1.weight * t1.value) ) as probability
from
  features_exploded t1
  LEFT OUTER JOIN classifier p1 
    ON (t1.feature = p1.feature)
group by
  t1.id
;
```

> #### Note
>
> `sigmoid()` should be applied only for logistic loss and you can't get a probability with other loss functions for a classification. See also [this video](https://www.coursera.org/lecture/machine-learning/decision-boundary-WuL1H).

Output for single sample can be:

|id| probability|
|---:|---:|
| 1| 1.0261879540562902e-10|

### Evaluation

If you have test samples for evaluation, use Hivemall's [evaluation UDFs](../eval/binary_classification_measures.html) to measure the accuracy of prediction.

For instance, prediction accuracy over the `training` samples can be measured as:

```sql
with features_exploded as (
  select
    id,
    extract_feature(fv) as feature,
    extract_weight(fv) as value
  from
    training t1 
    LATERAL VIEW explode(features) t2 as fv
),
predictions as (
  select
    t1.id,
    sigmoid( sum(p1.weight * t1.value) ) as probability
  from
    features_exploded t1
    LEFT OUTER JOIN classifier p1 
      ON (t1.feature = p1.feature)
  group by
    t1.id
)
select
  auc(probability, label) as auc,
  logloss(probability, label) as logloss
from (
  select 
    t1.probability, t2.label
  from 
    predictions t1
    join training t2 on (t1.id = t2.id)
  ORDER BY 
    probability DESC
) t
;
```

|auc|	logloss|
|---:|---:|
|0.5|	9.200000003614099|

Since we are trying to solve the binary classification problem, the accuracy is measured by [Area Under the ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) [`auc()`](../eval/auc.html) and/or [Logarithmic Loss](http://wiki.fast.ai/index.php/Log_Loss) [`logloss()`](../eval/regression.html#logarithmic-loss).

## Regression

If you use [`train_regressor()`](../misc/funcs.html#regression) instead of [`train_classifier()`](../misc/funcs.html#binary-classification), you can also solve a regression problem with almost same queries.

Imagine the following `customers` table:

```sql
create table if not exists customers as
select 1 as id, "male" as gender, 23 as age, "Japan" as country, 12 as num_purchases
union all
select 2 as id, "female" as gender, 43 as age, "US" as country, 4 as num_purchases
union all
select 3 as id, "other" as gender, 19 as age, "UK" as country, 2 as num_purchases
union all
select 4 as id, "male" as gender, 31 as age, "US" as country, 20 as num_purchases
union all
select 5 as id, "female" as gender, 37 as age, "Australia" as country, 9 as num_purchases
;
```

| gender | age | country | num_purchases |
|:---:|:---|:---:|:---|
| male | 23 |Japan | 12 |
| female | 43 | US | 4 |
| other | 19 | UK |  2 |
| male | 31 | US |  20 |
| female | 37 | Australia | 9 |

Now, our goal is to build a regression model to predict the number of purchases potentially done by new customers.

### Step 1. Feature representation

Same as the classification example:

```sql
insert overwrite table training
select
  id,
  array_concat(
    quantitative_features(
      array("age"),
      age
    ),
    categorical_features(
      array("country", "gender"),
      country, gender
    )
  ) as features,
  num_purchases
from
  customers
;
```

### Step 2. Training

[`train_regressor()`](../misc/funcs.html#regression) requires you to specify an appropriate loss function. One option is to replace the classifier-specific loss function `logloss` with `squared` as:

```sql
create table if not exists regressor as
select
  train_regressor(
    features, -- feature vector
    num_purchases, -- target value
    '-loss_function squared -optimizer AdaGrad' -- hyper-parameters
  ) as (feature, weight)
from
  training
;
```

`-loss_function squared` means that this query builds a simple linear regressor with the squared loss. Meanwhile, this example optimizes the parameters based on the `AdaGrad` optimization scheme with `l2` regularization.

Run the function with `-help` option to list available options:

```sql
select train_regressor('-help');
-- Hivemall 0.5.2 and before
-- select train_regressor(array(), 0, '-help');
```

### Step 3. Prediction

Prepare dummy new customers:

```sql
create table if not exists new_customers as
select 1 as id, array("gender#male", "age:10", "country#Japan") as features
union all
select 2 as id, array("gender#female", "age:60", "country#US") as features
union all
select 3 as id, array("gender#other", "age:50", "country#UK") as features
;
```

A way of prediction is almost the same as classification, but not need to pass through the [`sigmoid()`](../misc/generic_funcs.html#math) function:

```sql
with features_exploded as (
  select
    id,
    extract_feature(fv) as feature,
    extract_weight(fv) as value
  from new_customers t1 LATERAL VIEW explode(features) t2 as fv
)
select
  t1.id,
  sum(p1.weight * t1.value) as predicted_num_purchases
from
  features_exploded t1
  LEFT OUTER JOIN regressor p1 ON (t1.feature = p1.feature)
group by
  t1.id
;
```

Output is like:

|id| predicted\_num_purchases|
|---:|---:|
| 1| 3.645142912864685|

### Evaluation

Use [Root Mean Square Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) [`rmse()`](../misc/funcs.html#evaluation) or [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) [`mae()`](../misc/funcs.html#evaluation) UDFs for evaluation of regressors:

```sql
with features_exploded as (
  select
    id,
    extract_feature(fv) as feature,
    extract_weight(fv) as value
  from
    training t1 
    LATERAL VIEW explode(features) t2 as fv
),
predictions as (
  select
    t1.id,
    sum(p1.weight * t1.value) as predicted_num_purchases
  from
    features_exploded t1
    LEFT OUTER JOIN regressor p1 ON (t1.feature = p1.feature)
  group by
    t1.id
)
select
  rmse(t1.predicted_num_purchases, t2.num_purchases) as rmse,
  mae(t1.predicted_num_purchases, t2.num_purchases) as mae
from
  predictions t1
join
  training t2 on (t1.id = t2.id)
;
```

Output is like:

|rmse|mae|
|---:|---:|
|9.411633136764399|7.124141833186149|

## Next steps

See the following resources for further information:

- [Detailed documentation](./prediction.html) of `train_classifier` and `train_regressor`
  - Query examples for some public datasets are also available in it.
