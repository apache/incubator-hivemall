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

_Caution: SLIM is supported from Hivemall v0.xxx or later._

<!-- toc -->

# Data preparation

## Data binarization

In this article, each entry's value is binarized to reduce training samples and consider only high rated items, which rating is 4 or 5.
So, all matrix elements which have the lower rating than 4 aren't used during training.

```sql
SET hivevar:seed=31;

drop table ratings2;
create table ratings2 as
select
  rand(${seed}) as rnd,
  userid,
  movieid,
  1. as rating
from
  ratings
where rating >= 4.
;
```

`rnd` field is inserted into each record since the dataset is split into training and testing data.

## Splitting dataset

To evaluate a recommendation model, this tutorial uses two type cross validations:

- leave-one-out cross validation
- $$K$$-hold cross validation

The former is used in the [Slim's paper](http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf),
and the latter is used in [Mendely's slide](slideshare.net/MarkLevy/efficient-slides/).

### Leave-one-out cross validation

For leave-one-out cross validation,
the dataset is split into a training set and a testing set by randomly selecting one of the non-zero entries of each user and placing it into the testing set.

``` sql
drop table testing;
create table testing
as
WITH top_k as (
  select
     each_top_k(1, userid, rnd, userid, movieid)
      as (rank, rnd, userid, movieid)
  from (
    select
      *
    from
      ratings2
    CLUSTER BY
      userid
  ) t
)
select
  userid,
  movieid
from
  top_k
;

drop table training;
create table training as
select
  l.userid,
  l.movieid
from
  ratings2 l
where
  l.movieid not in
    (select
       r.movieid
    from
      testing r
    where
      l.userid=r.userid)
;
```

### $$K$$-hold corss validation case

When $$K=2$$, the dataset is divided into training data and testing dataset.
The number of samples training and testing dataset are roughly the same


```sql
drop table testing;
create table testing
as
select * from ratings2
    where rnd >= 0.5
;

drop table training;
create table training
as
select * from ratings2
    where rnd < 0.5
;
```

## Precompute movie-movie similarity

SLIM needs top-$$k$$ most similar items for each item to the approximate user-item matrix.
Here, we particularly focus on [DIMSUM](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation), an efficient and approximated similarity computation scheme.

Since we set `k=20`, the output has 20 most-similar movies per `movieid`.
We can adjust trade-off between training time and precision of matrix approximation by varying `k`.
Larger `k` is the better approximation for raw user-item matrix, but training time and memory usage tends to increase.

[As we explained in the general introduction of item-based CF](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation.md), following query finds top-$$k$$ nearest-neighborhood movies for each movie:

```sql
set hivevar:k=20;

DROP TABLE knn_train;
create table knn_train
as
with movie_magnitude as (
  select
    to_map(j, mag) as mags
  from (
    select
      movieid as j,
      l2_norm(rating) as mag
    from
      training
    group by
      movieid
  ) t0
),
movie_features as (
  select
    userid as i,
    collect_list(
      feature(movieid, rating)
    ) as feature_vector
  from
    training
  group by
    userid
),
partial_result as (
  select
    dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.1 -int_feature')
      as (movieid, other, s)
  from
    movie_features f
  left outer join movie_magnitude m
),
similarity as (
    select
      movieid,
      other,
      sum(s) as similarity
    from
      partial_result
    group by
      movieid, other
),
topk as (
  select
    each_top_k(
      ${k}, movieid, similarity, -- use top k items
      movieid, other
    ) as (rank, similarity, movieid, other)
  from (
    select * from similarity
    CLUSTER BY movieid
  ) t
)
select
  movieid, other, similarity
from
  topk
;
```

| movieid | other | similarity |
|:---:|:---:|:---|
| 1 | 3114 | 0.6344983426248486 |
| 1 | 1270 | 0.6040521649557099 |
| 1 | 1265 | 0.6016164707421789 |
| 1 | 551 | 0.5821309170339302 |
| 1 | 337 | 0.5796952228203991 |
|...|...|...|


> #### Caution
> To run query above, you may need to run two statement before.
```sql
set hive.strict.checks.cartesian.product=false;
set hive.mapred.mode=nonstrict;
```

## Create feature tables

Here, we create the feature table for training slim model.

Slim input features are :

- `i`: axis item id
- `Ri`: rating map of axis item. It has user id as key and rating as value.
- `j`: one of the top-$$K$$ similar item ids
- `Rj`: rating map of a similar item, which has user id as key and rating as value.
- `knn_i`: all top-$$K$$ similar items rating map: it has user id as key and map as value. map of value has item id as key and rating as value.: `map<userid, map<itemid, rating>>`

```sql
DROP TABLE item_matrix;
CREATE table item_matrix as
  select
    movieid as i,
    to_map(userid, rating) as R_i
  from
    training
  group by
    movieid;

-- Create SLIM input features
DROP TABLE slim_training_item;
CREATE TABLE slim_training_item as
with knn_item_user_matrix as (
  select
    l.movieid,
    r.userid,
    to_map(l.other, r.rating) ratings
  from
    knn_train l
    JOIN training r ON (l.other = r.movieid)
  group by l.movieid, r.userid
), knn_item_matrix as (
  select
    movieid as i,
    to_map(userid, ratings) as KNN_i -- map<userid, map<movieid, rating>>
  from
    knn_item_user_matrix
  group by
    movieid
)
select
  l.movieid,
  r1.R_i,
  r2.knn_i,
  l.other,
  r3.R_i as R_j
from
  knn_train l
    JOIN item_matrix r1 ON (l.movieid = r1.i)
    JOIN knn_item_matrix r2 ON (l.movieid = r2.i)
    JOIN item_matrix r3 ON (l.other = r3.i)
;
```

# Training

## Build a prediction model by SLIM

`slim_train` function outputs item-item non-negative sparse matrix.
For item recommendation, this matrix is stored into the table.

```sql
DROP TABLE slim_w;
create table slim_w as
select
  i, nn, avg(w) as w
from (
  select
    train_slim(i, r_i, knn_i, j, r_j) as (i, nn, w)
  from (
    select
      movieid as i,
      r_i,
      knn_i,
      other as j,
      r_j
    from slim_training_item
    CLUSTER BY i
  ) t1
) t2
group by i, nn
;
```

## Usage of `train_slim`

You can obtain information about `train_slim` and its arguments by giving `-help` option as follows:

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
                                     coordinate descent [default: 40]
 -l1,--l1coefficient <arg>           Coefficient for l1 regularizer
                                     [default: 0.001]
 -l2,--l2coefficient <arg>           Coefficient for l2 regularizer
                                     [default: 0.0005]
```


# Prediction unrated items for each user

Here, we predict values in non-one value elements in binarized user-item matrix exclude testing dataset based on training slim weights.
Non-one value element means a future movie rating by a user who has not seen this movie yet (or marked lower rating in this article's preprocessing).
If an element in the user-item matrix is predicted as high value,
we recommend the movie to the user since he or she is likely to highly evaluate it.

## Set hyperparameters

```sql
set hivevar:mu=0.;
```

> #### Caution
> When $$k$$ is small, slim predicted value may be `null`.
> `null` is replaced by `$mu`.

Mean value is also one good choice to replace `null` element.
Next, we predict rating of user-movie pairs based on top-$$K$$ similar items.

```sql
drop table predict_pair;
create table predict_pair
as
with cross_table as (
  select
    l.userid as u,
    t.movieid as m
  from
    testing l
  CROSS JOIN
    (select
       DISTINCT(movieid)
     from ratings
    ) t
)
select
  u as userid,
  m as movieid
from
  cross_table
where m not in (
  select
    movieid
  from
    training r
  where
    u=r.userid
);

DROP TABLE predicted_test_matrix;
CREATE TABLE predicted_test_matrix as
with knn_exploded as (
  select
    l.userid  as u,
    l.movieid as i, -- axis
    r1.other  as k, -- other
    r2.rating as r_uk
  from
    predict_pair l
  LEFT OUTER JOIN knn_train r1
    ON (l.movieid = r1.movieid)
  JOIN training r2
    ON (r1.other = r2.movieid and l.userid = r2.userid)
)
select
  l.u,
  l.i,
  coalesce(sum(l.r_uk*r.w), ${mu}) as predicted
from
  knn_exploded l
LEFT OUTER JOIN slim_w r
  ON (l.i = r.i and l.k = r.nn)
group by
  l.u, l.i
;
```

## Top-$$N$$ recommendation for each user

Here, we recommend top-`N=3` items for each user based on predicted values.

```sql
SET hivevar:n=3;

DROP TABLE if exists recommend;
create table recommend
as
WITH top_n as (
  select
     each_top_k(${n}, u, predicted, u, i)
      as (rank, predicted, userid, i)
  from (
    select
      *
    from
      predicted_test_matrix
    CLUSTER BY
      u
  ) t
)
select
  userid,
  collect_list(i) as items
from
  top_n
group by
  userid
;

select * from recommend;
```


| userid | items |
|:---:|:---:|
| 1 | [1246,588,150] |
| 2 | [1084,3256,1527] |
| 3 | [2858,1270,590] |
| 4 | [1954,3527] |
| 5 | [1213,1732,6] |
|...|...|


# Evaluation

## TOP-$$N$$ item evaluations: Hit Rate, ARHR and Precision@N

```sql
WITH top_k as (
  select
    each_top_k(10, u, predicted, u, i)
      as (rank, predicted, userid, i)
  from (
    select
      *
    from
      predicted_test_matrix
    CLUSTER BY u
  ) t
), rec_items as (
select
  userid,
  collect_list(i) as items
from
  top_k
group by
  userid
), truth_data as (
  select
    userid,
    collect_list(movieid) as truth
  from
    testing
  group by
    userid
) select
    hitrate(r.items, l.truth) as HITRATE,
    arhr(r.items, l.truth) as ARHR,
    precision_at(r.items, l.truth) as prec10
from
  truth_data l
join
  rec_items r on (l.userid=r.userid)
;
```

## MRR

``` sql
WITH ordered as (
  select
    u,
    i,
    predicted
  from
    predicted_test_matrix
  order by
    u, predicted desc
), rec_items as (
select
  u as userid,
  collect_list(i) as items
from
  ordered
group by
  u
), truth_data as (
select
  userid,
  collect_list(movieid) as truth
from
  testing
group by
  userid
)
select
  mrr(r.items, l.truth)
from
  truth_data l
join
  rec_items r on (l.userid=r.userid)
;
```
