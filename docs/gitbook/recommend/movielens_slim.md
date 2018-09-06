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

Hivemall supports a neighborhood-learning scheme using SLIM. 
SLIM is a representative of neighborhood-learning recommendation algorithm introduced in the following paper:

- Xia Ning and George Karypis, [SLIM: Sparse Linear Methods for Top-N Recommender Systems](https://dl.acm.org/citation.cfm?id=2118303), Proc. ICDM, 2011.

_Caution: SLIM is supported from Hivemall v0.5-rc.1 or later._

<!-- toc -->

# SLIM optimization objective

The optimization objective of [SLIM](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf) is similar to Elastic Net (L1+L2 regularization) with additional constraints as follows:

$$
\begin{aligned}
& \;{\tiny\begin{matrix}\\ \normalsize \text{minimize} \\ ^{\scriptsize w_{j}}\end{matrix}}\; 
&& \frac{1}{2}\Vert r_{j} - Rw_{j} \Vert_2^2 + \frac{\beta}{2} \Vert w_{j} \Vert_2^2 + \lambda \Vert w_{j} \Vert_1 \\
& \text{subject to} 
&& w_{j} \geq 0 \\
&&& diag(W)= 0
\end{aligned}
$$

# Data preparation

## Rating binarization

In this article, each user-movie matrix element is binarized to reduce training samples and consider only high rated movies whose rating is 4 or 5. So, every matrix element having a lower rating than 4 is not used for training.

```sql
SET hivevar:seed=31;

DROP TABLE ratings2;
CREATE TABLE ratings2 as
select
  rand(${seed}) as rnd,
  userid,
  movieid as itemid,
  cast(1.0 as float) as rating -- double is also accepted
from
  ratings
where rating >= 4.
;
```

`rnd` field is appended for each record to split `ratings2` into training and testing data later.

Binarization is an optional step, and you can use raw rating values to train a SLIM model.

## Splitting dataset

To evaluate a recommendation model, this tutorial uses two type cross validations:

- Leave-one-out cross validation
- $$K$$-hold cross validation

The former is used in the [SLIM's paper](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf) and the latter is used in [Mendeley's slide](https://www.slideshare.net/MarkLevy/efficient-slides/).

### Leave-one-out cross validation

For leave-one-out cross validation, the dataset is split into a training set and a testing set by randomly selecting one of the non-zero entries of each user and placing it into the testing set.
In the following query, the movie has the smallest `rnd` value is used as test data (`testing` table) per a user.
And, the others are used as training data (`training` table).

When we select slim's best hyperparameters, different test data is used in [evaluation section](#evaluation) several times.

``` sql
DROP TABLE testing;
CREATE TABLE testing
as
WITH top_k as (
  select
     each_top_k(1, userid, rnd, userid, itemid, rating)
      as (rank, rnd, userid, itemid, rating)
  from (
    select * from ratings2
    CLUSTER BY userid
  ) t
)
select
  userid, itemid, rating
from
  top_k
;

DROP TABLE training;
CREATE TABLE training as
select
  l.*
from
  ratings2 l
  LEFT OUTER JOIN testing r ON (l.userid=r.userid and l.itemid=r.itemid)
where
  r.itemid IS NULL -- anti join
;
```

### $$K$$-hold corss validation

When $$K=2$$, the dataset is divided into training data and testing dataset.
The numbers of training and testing samples roughly equal.

When we select slim's best hyperparameters, you'll first train a SLIM prediction model from training data and evaluate the prediction model by testing data.

Optionally, you can switch training data with testing data and evaluate again.

```sql
DROP TABLE testing;
CREATE TABLE testing
as
select * from ratings2
where rnd >= 0.5
;

DROP TABLE training;
CREATE TABLE training
as
select * from ratings2
where rnd < 0.5
;
```

> #### Note
>
> In the following section excluding evaluation section,
> we will show the example of queries and its results based on $$K$$-hold cross validation case.
> But, this article's queries are valid for leave-one-out cross validation.

## Pre-compute item-item similarity

SLIM needs top-$$k$$ most similar movies for each movie to the approximate user-item matrix.
Here, we particularly focus on [DIMSUM](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation),
an efficient and approximated similarity computation scheme.

Because we set `k=20`, the output has 20 most-similar movies per `itemid`.
We can adjust trade-off between training and prediction time and precision of matrix approximation by varying `k`.
Larger `k` is the better approximation for raw user-item matrix, but training time and memory usage tend to increase.

[As we explained in the general introduction of item-based CF](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation.md),
following query finds top-$$k$$ nearest-neighborhood movies for each movie:

```sql
set hivevar:k=20;

DROP TABLE knn_train;
CREATE TABLE knn_train
as
with item_magnitude as (
  select
    to_map(j, mag) as mags
  from (
    select
      itemid as j,
      l2_norm(rating) as mag
    from
      training
    group by
      itemid
  ) t0
),
item_features as (
  select
    userid as i,
    collect_list(
      feature(itemid, rating)
    ) as feature_vector
  from
    training
  group by
    userid
),
partial_result as (
  select
    dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.1 -int_feature')
      as (itemid, other, s)
  from
    item_features f
    CROSS JOIN item_magnitude m
),
similarity as (
  select
    itemid,
    other,
    sum(s) as similarity
  from
    partial_result
  group by
    itemid, other
),
topk as (
  select
    each_top_k(
      ${k}, itemid, similarity, -- use top k items
      itemid, other
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    CLUSTER BY itemid
  ) t
)
select
  itemid, other, similarity
from
  topk
;
```

| itemid | other | similarity |
|:---:|:---:|:---|
| 1 | 3114 | 0.28432244 |
| 1 | 1265 | 0.25180137 |
| 1 | 2355 | 0.24781825 |
| 1 | 2396 | 0.24435896 |
| 1 | 588  | 0.24359442 |
|...|...|...|


> #### Caution
> To run the query above, you may need to run the following statements:
```sql
set hive.strict.checks.cartesian.product=false;
set hive.mapred.mode=nonstrict;
```

## Create training input tables

Here, we prepare input tables for SLIM training.

SLIM input consists of the following columns in `slim_training_item`:

- `i`: axis item id
- `Ri`: the user-rating vector of the axis item $$i$$ expressed as `map<userid, rating>`.
- `knn_i`: top-$$K$$ similar item matrix of item $$i$$; the user-item rating matrix is expressed as `map<userid, map<itemid, rating>>`.
- `j`: an item id in `knn_i`.
- `Rj`: the user-rating vector of the item $$j$$ expressed as `map<userid, rating>`.

```sql
DROP TABLE item_matrix;
CREATE table item_matrix as
select
  itemid as i,
  to_map(userid, rating) as R_i
from
  training
group by
  itemid;

-- Temporary set off map join because the following query does not work well for map join
set hive.auto.convert.join=false;
-- set mapred.reduce.tasks=64;

-- Create SLIM input features
DROP TABLE slim_training_item;
CREATE TABLE slim_training_item as
WITH knn_item_user_matrix as (
  select
    l.itemid,
    r.userid,
    to_map(l.other, r.rating) ratings
  from
    knn_train l
    JOIN training r ON (l.other = r.itemid)
  group by
    l.itemid, r.userid
),
knn_item_matrix as (
  select
    itemid as i,
    to_map(userid, ratings) as KNN_i -- map<userid, map<itemid, rating>>
  from
    knn_item_user_matrix
  group by
    itemid
)
select
  l.itemid as i,
  r1.R_i,
  r2.knn_i,
  l.other as j,
  r3.R_i as R_j
from
  knn_train l
  JOIN item_matrix r1 ON (l.itemid = r1.i)
  JOIN knn_item_matrix r2 ON (l.itemid = r2.i)
  JOIN item_matrix r3 ON (l.other = r3.i)
;

-- set to the default value
set hive.auto.convert.join=true;
```

# Training

## Build a prediction model by SLIM

`train_slim` function outputs the nonzero elements of an item-item matrix.
For item recommendation or prediction, this matrix is stored into the table named `slim_model`.

```sql
DROP TABLE slim_model;
CREATE TABLE slim_model as
select
  i, nn, avg(w) as w
from (
  select
    train_slim(i, r_i, knn_i, j, r_j) as (i, nn, w)
  from (
    select * from slim_training_item
    CLUSTER BY i
  ) t1
) t2
group by i, nn
;
```

## Usage of `train_slim`

You can obtain information about `train_slim` function and its arguments by giving `-help` option as follows:

``` sql
select train_slim("-help");
```

``` sql
usage: train_slim( int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI,
       int j, map<int, double> r_j [, constant string options])
       - Returns row index, column index and non-zero weight value of prediction model
       [-cv_rate <arg>] [-disable_cv] [-help] [-iters <arg>] [-l1 <arg>] [-l2 <arg>]
 -cv_rate,--convergence_rate <arg>   Threshold to determine convergence
                                     [default: 0.005]
 -disable_cv,--disable_cvtest        Whether to disable convergence check
                                     [default: enabled]
 -help                               Show function help
 -iters,--iterations <arg>           The number of iterations for
                                     coordinate descent [default: 30]
 -l1,--l1coefficient <arg>           Coefficient for l1 regularizer
                                     [default: 0.001]
 -l2,--l2coefficient <arg>           Coefficient for l2 regularizer
                                     [default: 0.0005]
```

# Prediction and recommendation

Here, we predict ratng values of binarized user-item rating matrix of testing dataset based on ratings in training dataset.

Based on predicted rating scores, we can recommend top-k items for each user that he or she will be likely to put high scores.

## Predict unknown ratings of a user-item matrix

Based on known ratings and SLIM weight matrix, we predict unknown ratings in the user-item matrix.
SLIM predicts ratings of user-item pairs based on top-$$K$$ similar items.

The `predict_pair` table represents candidates for recommended user-movie pairs, excluding known ratings in the training dataset.

```sql
CREATE OR REPLACE VIEW predict_pair 
as
WITH testing_users as (
  select DISTINCT(userid) as userid from testing
),
training_items as (
  select DISTINCT(itemid) as itemid from training
),
user_items as (
  select
    l.userid,
    r.itemid
  from
    testing_users l
    CROSS JOIN training_items r
)
select
  l.userid,
  l.itemid
from
  user_items l
  LEFT OUTER JOIN training r ON (l.userid=r.userid and l.itemid=r.itemid)
where
  r.itemid IS NULL -- anti join
;
```

```sql
-- optionally set the mean/default value of prediction
set hivevar:mu=0.0;

DROP TABLE predicted;
CREATE TABLE predicted 
as
WITH knn_exploded as (
  select
    l.userid  as u,
    l.itemid as i, -- axis
    r1.other  as k, -- other
    r2.rating as r_uk
  from
    predict_pair l
    LEFT OUTER JOIN knn_train r1
      ON (r1.itemid = l.itemid)
    JOIN training r2
      ON (r2.userid = l.userid and r2.itemid = r1.other)
)
select
  l.u as userid,
  l.i as itemid,
  coalesce(sum(l.r_uk * r.w), ${mu}) as predicted
  -- coalesce(sum(l.r_uk * r.w)) as predicted
from
  knn_exploded l
  LEFT OUTER JOIN slim_model r ON (l.i = r.i and l.k = r.nn)
group by
  l.u, l.i
;
```

> #### Caution
> When $$k$$ is small, slim predicted value may be `null`. Then, `$mu` replaces `null` value.
> The mean value of item ratings is a good choice for `$mu`.

## Top-$$K$$ item recommendation for each user

Here, we recommend top-3 items for each user based on predicted values.

```sql
SET hivevar:k=3;

DROP TABLE IF EXISTS recommend;
CREATE TABLE recommend
as
WITH top_n as (
  select
     each_top_k(${k}, userid, predicted, userid, itemid)
      as (rank, predicted, userid, itemid)
  from (
    select * from predicted
    CLUSTER BY userid
  ) t
)
select
  userid,
  collect_list(itemid) as items
from
  top_n
group by
  userid
;

select * from recommend limit 5;
```

| userid | items |
|:---:|:---:|
| 1 | [364,594,2081] |
| 2 | [2028,3256,589] |
| 3 | [260,1291,2791] |
| 4 | [1196,1200,1210] |
| 5 | [3813,1366,89] |
|...|...|

# Evaluation

## Top-$$K$$ ranking measures: Hit-Rate@K, MRR@K, and Precision@K

In this section, `Hit-Rate@k`, `MRR@k`, and `Precision@k` are computed based on recommended items.

[`Precision@K`](../eval/rank.html#precision-at-k) is a good evaluation measure for $$K$$-hold cross validation.

On the other hand, `Hit-Rate` and [`Mean Reciprocal Rank`](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) (i.e., Average Reciprocal Hit-Rate) are good evaluation measures for leave-one-out cross validation.

```sql
SET hivevar:n=10;

WITH top_k as (
  select
    each_top_k(${n}, userid, predicted, userid, itemid)
      as (rank, predicted, userid, itemid)
  from (
    select * from predicted
    CLUSTER BY userid
  ) t
),
rec_items as (
  select
    userid,
    collect_list(itemid) as items
  from
    top_k
  group by
    userid
),
ground_truth as (
  select
    userid,
    collect_list(itemid) as truth
  from
    testing
  group by
    userid
)
select
  hitrate(l.items, r.truth) as hitrate,
  mrr(l.items, r.truth) as mrr,
  precision_at(l.items, r.truth) as prec
from
  rec_items l
  join ground_truth r on (l.userid=r.userid)
;
```

### Leave-one-out result

| hitrate | mrr | prec |
|:-------:|:---:|:----:|
| 0.21517309922146763 | 0.09377752536606271 | 0.021517309922146725 |

Hit Rate and MRR are similar to ones in [the result of Table II in Slim's paper](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf)

### $$K$$-hold result

| hitrate | mrr | prec |
|:-------:|:---:|:----:|
| 0.8952775476387739 | 1.1751514972186057 | 0.3564871582435789 |

Precision value is similar to [the result of Mendeley's slide](https://www.slideshare.net/MarkLevy/efficient-slides/13).

## Ranking measures: MRR

In this example, whole recommended items are evaluated using MRR.

``` sql
WITH rec_items as (
  select
    userid,
    to_ordered_list(itemid, predicted, '-reverse') as items
  from
    predicted
  group by
    userid
),
ground_truth as (
  select
    userid,
    collect_list(itemid) as truth
  from
    testing
  group by
    userid
)
select
  mrr(l.items, r.truth) as mrr
from
  rec_items l
  join ground_truth r on (l.userid=r.userid)
;
```

### Leave-one-out result

| mrr |
|:---:|
| 0.10782647321821472 |

### $$K$$-hold result

| mrr |
|:---:|
| 0.6179983058881773 |

This MRR value is similar to one in [the Mendeley's slide](https://www.slideshare.net/MarkLevy/efficient-slides/13).
