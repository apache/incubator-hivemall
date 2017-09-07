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

First of all, please create `ratings`, `training` and `testing` tables described in [this article](../recommend/movielens_dataset.html).
Here, we use `training` data to build prediction model, and `testing` data to recommend items.

## Precompute movie-movie similarity

SLIM needs top-$$k$$ most similar items for each item to approximate user-item matrix.
Here, we particularly focus on [DIMSUM](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation), an efficient and approximated similarity computation scheme, and try to make recommendation from the MovieLens data.

Since we set `k=20`, output has 20 most-similar movies per `movieid`.
We can adjust trade-off between training time and approximation by varying `k`.
Larger `k` is better approximation for raw user-item matrix, but training time increases.

[As we explained in the general introduction of item-based CF](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation.md), following query finds top-$$k$$ nearest-neighborhood movies for each movie:

```sql
set hivevar:k=30;

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
    to_map(userid, ratings) as KNN_i -- map<userid, map<movieid,rating>>
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

## Set hyperparamters

```sql
-- mean rating
set hivevar:mu=0.;
```

> #### Caution
> When $$k$$ is small, slim predicted value may be `null`.
> `null` is replaced `$mu`.


## Build a prediction model by SLIM

```sql
DROP TABLE slim_w;
create table slim_w as
select
  i, j, avg(wij) as wij
from (
  select
    train_slim(i, r_i, knn_i, j, r_j) as (i, j, wij)
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
group by i, j
;
```

## Usage of `train_slim`

# Prediction

Next, we predict rating of user-movie pairs based on top-$$k$$ similar items:

```sql
DROP TABLE predicted_test_matrix;
CREATE TABLE predicted_test_matrix as
with knn_exploded as (
  select
    l.userid  as u,
    l.movieid as i, -- axis 
    r1.other   as k, -- other
    r2.rating  as r_uk
  from testing l
  LEFT OUTER JOIN knn_train r1
    ON (l.movieid = r1.movieid)
  JOIN training r2 
    ON (r1.other = r2.movieid and l.userid = r2.userid)
)
select 
  l.u,
  l.i,
  coalesce(sum(l.r_uk*r.wij), ${mu}) as predicted
from knn_exploded l
LEFT OUTER JOIN slim_w r 
  ON (l.i = r.i and l.k = r.j)
group by l.u, l.i
;
```

> #### Note
> When prediction for test data, we can use top-$$k$$ similarity for all data: training and testing data.
> If you want to do that, please run query to compute top-$$k$$ item-item similarity on all data.



# Top-3 recommendation for each user

```sql
DROP TABLE if exists top3_recommend;
create table top3_recommend 
as 
WITH top_k as (
  select
     each_top_k(3, u, predicted, u, i)
      as (rank, predicted, userid, i)
  from (
    select * from predicted_test_matrix
    CLUSTER BY u
  ) t
)
select
  userid,
  collect_list(i) as items
from
  top_k
group by
  userid
;

select * from top3_recommend;
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
