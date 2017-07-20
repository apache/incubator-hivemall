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

[Our user guide for item-based collaborative filtering (CF)](item_based_cf.md) introduced how to make recommendation based on item-item similarities. Here, we particularly focus on [DIMSUM](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation), an efficient and approximated similarity computation scheme, and try to make recommendation from the MovieLens data.

<!-- toc -->

# Compute movie-movie similarity

[As we explained in the general introduction of item-based CF](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation.md), following query finds top-$$k$$ nearest-neighborhood movies for each movie:

```sql
drop table if exists dimsum_movie_similarity;
create table dimsum_movie_similarity 
as
with movie_magnitude as ( -- compute magnitude of each movie vector
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
partial_result as ( -- launch DIMSUM in a MapReduce fashion
  select
    dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.1 -disable_symmetric_output')
      as (movieid, other, s)
  from
    movie_features f
  left outer join movie_magnitude m
),
similarity as ( -- reduce (i.e., sum up) mappers' partial results
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
    each_top_k( -- get top-10 nearest neighbors based on similarity score
      10, movieid, similarity,
      movieid, other -- output items
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
|1    |   2095 |   0.9377422722094696 |
|1    |   231  |   0.9316530366756418 |
|1    |   1407 |   0.9194745656079863 |
|1    |   3442 |   0.9133853300741587 |
|1    |   1792 |   0.9072960945403309 |
|...|...|...|

Since we set `k=10`, output has 10 most-similar movies per `movieid`.

> #### Note
> Since we specified an option `-disable_symmetric_output`, output table does not contain inverted similarities such as `<2095, 1>`, `<231, 1>`, `<1407, 1>`, ...

# Prediction

Next, we predict rating for unforeseen user-movie pairs based on the top-$$k$$ similarities:

```sql
drop table if exists dimsum_prediction;
create table dimsum_prediction
as
with similarity_all as (
  -- copy (i1, i2)'s similarity as (i2, i1)'s one
  select movieid, other, similarity from dimsum_movie_similarity
  union all
  select other as movieid, movieid as other, similarity from dimsum_movie_similarity
)
select 
	-- target user
	t1.userid,
	
	-- recommendation candidate
	t2.movieid,
	
	-- predicted rating: r_{u,i} = sum(s_{i,:} * r_{u,:}) / sum(s_{i,:})
	sum(t1.rating * t2.similarity) / sum(t2.similarity) as rating
from
	training t1 -- r_{u,<movieid>}
left join -- s_{i,<other>}
	similarity_all t2 
	on t1.movieid = t2.other
where
	-- do not include movies that user already rated
	NOT EXISTS (
		SELECT a.movieid FROM training a
		WHERE a.userid = t1.userid AND a.movieid = t2.movieid
	)
group by
	t1.userid, t2.movieid
;
```

This query computes estimated rating as follows:

|userid|movieid|rating|
|:---:|:---:|:---|
|1  |     1000  |  5.0 |
|1  |     1010  |  5.0 |
|1  |     1012  |  4.246349332667371 |
|1  |     1013  |  5.0 |
|1  |     1014  |  5.0 |
| ... | ... | ... |

Theoretically, for the $$t$$-th nearest-neighborhood item $$\tau(t)$$, prediction can be done by top-$$k$$ weighted sum of target user's historical ratings:
$$
r_{u,i} = \frac{\sum^k_{t=1} s_{i,\tau(t)} \cdot r_{u,\tau(t)} }{ \sum^k_{t=1} s_{i,\tau(t)} },
$$
where $$r_{u,i}$$ is user $$u$$'s rating for item (movie) $$i$$, and $$s_{i,j}$$ is similarity of $$i$$-$$j$$ (`movieid`-`other`) pair.

> #### Caution
> Since the number of similarities and users' past ratings are limited, we cannot say this output **always** contains prediction for **every** unforeseen user-item pairs; sometimes prediction for a specific user-item pair might be missing (i.e., `NULL`).

In fact, our goal is to make recommendation, but we can evaluate the intermediate result as a rating prediction problem:

```sql
select
	mae(t1.rating, t2.rating) as mae,
	rmse(t1.rating, t2.rating) as rmse
from
	testing t1 
left join
	dimsum_prediction t2
	on t1.movieid = t2.movieid
where
	t1.userid = t2.userid
;
```

| mae  |    rmse |
|:---:|:---:|
|0.7308365821230256   |   0.9594799959251938 |

Rating of the MovieLens data is in `[1, 5]` range, so this average errors are reasonable as a predictor.

# Recommendation

By using the prediction table, making recommendation for each user is straightforward:

```sql
drop table if exists dimsum_recommendation;
create table dimsum_recommendation
as
select
  userid,
  map_values(to_ordered_map(rating, movieid, true)) as rec_movies
from
  dimsum_prediction
group by
  userid
;
```

|userid  |  rec_movies |
|:---:|:---|
|1      | ["2590","999","372","1380","2078",...]
|2      | ["580","945","43","36","1704",...]
|3      | ["3744","852","1610","3740","2915",...]
|4      | ["3379","923","1997","2194","2944",...]
|5      | ["998","101","2696","2968","2275",...] |
| ... | ... |

> #### Note
> Size of `rec_movies` varies depending on each user's `training` samples and what movies he/she already rated. 

# Evaluation

Eventually, you can measure the quality of recommendation by using [ranking measures](../eval/rank.md):

```sql
with truth as (
  select 
    userid, 
    map_values(to_ordered_map(rating, cast(movieid as string), true)) as truth
  from
    testing
  group by
    userid
)
select 
  recall(t1.rec_movies, t2.truth, 10) as recall,
  precision(t1.rec_movies, t2.truth, 10) as precision,
  average_precision(t1.rec_movies, t2.truth) as average_precision,
  auc(t1.rec_movies, t2.truth) as auc,
  mrr(t1.rec_movies, t2.truth) as mrr,
  ndcg(t1.rec_movies, t2.truth) as ndcg
from
  dimsum_recommendation t1
join
  truth t2 on t1.userid = t2.userid
where -- at least 10 recommended items are necessary to compute recall@10 and precision@10
  size(t1.rec_movies) >= 10
;
```

| measure | accuracy  |
|:---:|:---|
|**Recall@10**| 0.027033598585322713  | 
|**Precision@10**| 0.009001989389920506  | 
|**Average Precision** | 0.017363681149831108  | 
|**AUC**| 0.5264553136097863   |  
|**MRR**| 0.03507380742291146   | 
|**NDCG**| 0.15787655209987522 |

If you set larger value to the DIMSUM's `-threshold` option, similarity will be more aggressively approximated. Consequently, while efficiency is improved, the accuracy is likely to be decreased.
