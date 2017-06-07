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
        
This document describes how to do Item-based Collaborative Filtering using Hivemall.

<!-- toc -->

> #### Caution
> Naive similarity computation is `O(n^2)` to compute all item-item pair similarity. In order to accelerate the procedure, Hivemall has an efficient scheme for computing Jaccard and/or cosine similarity [as mentioned later](#efficient-similarity-computation).

# Prepare `transaction` table

Prepare following `transaction` table. We will generate `feature_vector` for each `itemid` based on cooccurrence of purchased items, a sort of bucket analysis.

| userid | itemid | purchase_at `timestamp` |
|:-:|:-:|:-:| 
| 1 | 31231 | 2015-04-9 00:29:02 |
| 1 | 13212 | 2016-05-24 16:29:02 |
| 2 | 312 | 2016-06-03 23:29:02 |
| 3 | 2313 | 2016-06-04 19:29:02 |
| .. | .. | .. |

# Create `item_features` table

What we want for creating a feature vector for each item is the following `cooccurrence` relation.

| itemid | other | cnt |
|:-:|:-:|:-:|
| 583266 | 621056 | 9999 |
| 583266 | 583266 | 18 |
| 31231 | 13212 | 129 |
| 31231 | 31231 | 3 |
| 31231	| 9833 | 953 |
| ... | ... | ... |

Feature vectors of each item will be as follows:

| itemid | feature_vector `array<string>` |
|:-:|:-|
| 583266 | 621056:9999, 583266:18 |
| 31231 | 13212:129, 31231:3, 9833:953 |
| ... | ... |

Note that value of feature vector should be scaled for k-NN similarity computation e.g., as follows:

| itemid | feature_vector `array<string>` |
|:-:|:-|
| 583266 | 621056:`ln(9999+1)`, 583266:`ln(18+1)` |
| 31231 | 13212:`ln(129+1)`, 31231:`ln(3+1)`, 9833:`ln(953+1)` |
| ... | ... |

The following queries results in creating the above table.

## Step 1: Creating `user_purchased` table

The following query creates a table that contains `userid`, `itemid`, and `purchased_at`. The table represents the last user-item contact (purchase) while the `transaction` table holds all contacts.

```sql
CREATE TABLE user_purchased as
-- INSERT OVERWRITE TABLE user_purchased
select 
  userid,
  itemid,
  max(purchased_at) as purchased_at,
  count(1) as purchase_count
from
  transaction
-- where purchased_at < xxx -- divide training/testing data by time 
group by
  userid, itemid
;
```

> #### Note
> Better to avoid too old transactions because those information would be outdated though an enough number of transactions is required for recommendation.

## Step 2: Creating `cooccurrence` table

> #### Caution
> Item-item cooccurrence matrix is a symmetric matrix that has the number of total occurrence for each diagonal element. If the size of items is `k`, then the size of expected matrix is `k * (k - 1) / 2`, usually a very large one. Hence, it is better to use [step 2-2](#step-2-2-create-cooccurrence-table-from-upper-triangular-matrix-of-cooccurrence) instead of [step 2-1](#step-2-1-create-cooccurrence-table-directly) for creating a `cooccurrence` table where dataset is large.

### Step 2-1: Create `cooccurrence` table directly

```sql
create table cooccurrence as 
-- INSERT OVERWRITE TABLE cooccurrence
select
  u1.itemid,
  u2.itemid as other, 
  count(1) as cnt
from
  user_purchased u1
  JOIN user_purchased u2 ON (u1.userid = u2.userid)
where
  u1.itemid != u2.itemid 
  -- AND u2.purchased_at >= u1.purchased_at -- the other item should be purchased with/after the base item
group by
  u1.itemid, u2.itemid
-- having -- optional but recommended to have this condition where dataset is large
--  cnt >= 2 -- count(1) >= 2
;
```

> #### Note 
> Note that specifying `having cnt >= 2` has a drawback that item cooccurrence is not calculated where `cnt` is less than 2. It could result no recommended items for certain items. Please ignore `having cnt >= 2` if the following computations finish in an acceptable/reasonable time.

<br/>

> #### Caution
> We ignore a purchase order in the following example. It means that the occurrence counts of `ItemA -> ItemB` and `ItemB -> ItemA` are assumed to be same. It is sometimes not a good idea in terms of reasoning; for example, `Camera -> SD card` and `SD card -> Camera` need to be considered separately.

### Step 2-2: Create `cooccurrence` table from Upper Triangular Matrix of cooccurrence

Better to create [Upper Triangular Matrix](https://en.wikipedia.org/wiki/Triangular_matrix#Description) that has `itemid > other` if resulting table is very large. No need to create Upper Triangular Matrix if your Hadoop cluster can handle the following instructions without considering it.

```sql
create table cooccurrence_upper_triangular as 
-- INSERT OVERWRITE TABLE cooccurrence_upper_triangular
select
  u1.itemid,
  u2.itemid as other, 
  count(1) as cnt
from
  user_purchased u1
  JOIN user_purchased u2 ON (u1.userid = u2.userid)
where
  u1.itemid > u2.itemid 
group by
  u1.itemid, u2.itemid
;
```

```sql
create table cooccurrence as 
-- INSERT OVERWRITE TABLE cooccurrence
select * from (
  select itemid, other, cnt from cooccurrence_upper_triangular
  UNION ALL
  select other as itemid, itemid as other, cnt from cooccurrence_upper_triangular
) t; 
```

> #### Note
> `UNION ALL` [required to be embedded](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+Union#LanguageManualUnion-UNIONwithinaFROMClause) in Hive.

#### Limiting size of elements in `cooccurrence_upper_triangular`

Using only top-N frequently co-occurred item pairs allows you to reduce the size of `cooccurrence` table:

```sql
create table cooccurrence_upper_triangular as
WITH t1 as (
  select
    u1.itemid,
    u2.itemid as other, 
    count(1) as cnt
  from
    user_purchased u1
    JOIN user_purchased u2 ON (u1.userid = u2.userid)
  where
    u1.itemid > u2.itemid 
  group by
    u1.itemid, u2.itemid
),
t2 as (
  select
    each_top_k( -- top 1000
      1000, itemid, cnt, 
      itemid, other, cnt
    ) as (rank, cmpkey, itemid, other, cnt)
  from (
    select * from t1
    CLUSTER BY itemid
  ) t;
)
-- INSERT OVERWRITE TABLE cooccurrence_upper_triangular
select itemid, other, cnt
from t2;
```

```sql
create table cooccurrence as 
WITh t1 as (
  select itemid, other, cnt from cooccurrence_upper_triangular
  UNION ALL
  select other as itemid, itemid as other, cnt from cooccurrence_upper_triangular
),
t2 as (
  select
    each_top_k(
      1000, itemid, cnt,
      itemid, other, cnt
    ) as (rank, cmpkey, itemid, other, cnt)
  from (
    select * from t1
    CLUSTER BY itemid
  ) t
)
-- INSERT OVERWRITE TABLE cooccurrence
select itemid, other, cnt
from t2;
```

### Computing cooccurrence ratio (optional step)

You can optionally compute cooccurrence ratio as follows:

```sql
WITH stats as (
  select 
    itemid,
    sum(cnt) as totalcnt
  from 
    cooccurrence
  group by
    itemid
)
INSERT OVERWRITE TABLE cooccurrence_ratio
SELECT
  l.itemid,
  l.other, 
  (l.cnt / r.totalcnt) as ratio
FROM
  cooccurrence l
  JOIN stats r ON (l.itemid = r.itemid)
group by
  l.itemid, l.other
;
```

`l.cnt / r.totalcnt` represents a cooccurrence ratio of range `[0,1]`.

## Step 3: Creating a feature vector for each item

```sql
INSERT OVERWRITE TABLE item_features
SELECT
  itemid,
  -- scaling `ln(cnt+1)` to avoid large value in the feature vector
  -- rounding to xxx.yyyyy to reduce size of feature_vector in array<string>
  collect_list(feature(other, round(ln(cnt+1),5))) as feature_vector
FROM
  cooccurrence
GROUP BY
  itemid
;
```

# Compute item similarity scores

Item-item similarity computation is known to be computational complexity `O(n^2)` where `n` is the number of items. We have two options to compute the similarities, and, depending on your cluster size and your dataset, the optimal solution differs. 

> #### Note 
> If your dataset is large enough, better to choose [modified version of option 1](#taking-advantage-of-the-symmetric-property-of-item-similarity-matrix), which utilizes the symmetric property of similarity matrix.

## Option 1: Parallel computation with computationally heavy shuffling

This version involves 3-way joins w/ large data shuffle; However, this version works in parallel where a cluster has enough task slots.

```sql
WITH similarity as (
  select
    o.itemid,
    o.other,
    cosine_similarity(t1.feature_vector, t2.feature_vector) as similarity
  from
    cooccurrence o
    JOIN item_features t1 ON (o.itemid = t1.itemid)
    JOIN item_features t2 ON (o.other = t2.itemid)
),
topk as (
  select
    each_top_k( -- get top-10 items based on similarity score
      10, itemid, similarity,
      itemid, other -- output items
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    where similarity > 0 -- similarity > 0.01
    CLUSTER BY itemid
  ) t
)
INSERT OVERWRITE TABLE item_similarity
select 
  itemid, other, similarity
from 
  topk;
```

### Taking advantage of the symmetric property of item similarity matrix

Notice that `item_similarity` is a symmetric matrix. So, you can compute it from an upper triangular matrix as follows.

```sql
WITH cooccurrence_top100 as (
  select
    each_top_k(
      100, itemid, cnt,  
      itemid, other
    ) as (rank, cmpkey, itemid, other)
  from (
    select * from cooccurrence_upper_triangular
    CLUSTER BY itemid
  ) t
), 
similarity as (
  select
    o.itemid,
    o.other,
    cosine_similarity(t1.feature_vector, t2.feature_vector) as similarity
  from
    cooccurrence_top100 o　
    -- cooccurrence_upper_triangular o
    JOIN item_features t1 ON (o.itemid = t1.itemid)
    JOIN item_features t2 ON (o.other = t2.itemid)
),
topk as (
  select
    each_top_k( -- get top-10 items based on similarity score
      10, itemid, similarity,
      itemid, other -- output items
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    where similarity > 0 -- similarity > 0.01
    CLUSTER BY itemid
  ) t
)
INSERT OVERWRITE TABLE item_similarity_upper_triangler
select 
  itemid, other, similarity
from 
  topk;
```

```sql
INSERT OVERWRITE TABLE item_similarity
select * from (
  select itemid, other, similarity from item_similarity_upper_triangler
  UNION ALL
  select other as itemid, itemid as other, similarity from item_similarity_upper_triangler
) t;
```

## Option 2: Sequential computation

Alternatively, you can compute cosine similarity as follows. This version involves cross join and thus runs sequentially in a single task. However, it involves less shuffle compared to option 1.

```sql
WITH similarity as (
  select
   t1.itemid,
   t2.itemid as other,
   cosine_similarity(t1.feature_vector, t2.feature_vector) as similarity
  from
   item_features t1
   CROSS JOIN item_features t2
  WHERE
    t1.itemid != t2.itemid
),
topk as (
  select
    each_top_k( -- get top-10 items based on similarity score
      10, itemid, similarity,
      itemid, other -- output items
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    where similarity > 0 -- similarity > 0.01
    CLUSTER BY itemid
  ) t
)
INSERT OVERWRITE TABLE item_similarity
select 
  itemid, other, similarity
from 
  topk
;
```

| item | other | similarity |
|:-:|:-:|:-:|
| 583266 | 621056 | 0.33 |
| 583266 | 583266 | 0.18 |
| 31231 | 13212 | 1.29 |
| 31231 | 31231 | 0.3 |
| 31231	| 9833 | 0.953 |
| ... | ... | ... |

# Item-based recommendation

This section introduces item-based recommendation based on recently purchased items by each user.

> #### Caution
> It would better to ignore recommending some of items that user already purchased (only 1 time) while items that are purchased twice or more would be okey to be included in the recommendation list (e.g., repeatedly purchased daily necessities). So, you would need an item property table showing that each item is repeatedly purchased items or not.

## Step 1: Computes top-k recently purchased items for each user

First, prepare `recently_purchased_items` table as follows:

```sql
INSERT OVERWRITE TABLE recently_purchased_items
select
  each_top_k( -- get top-5 recently purchased items for each user
     5, userid, purchased_at,
     userid, itemid
  ) as (rank, purchased_at, userid, itemid)
from (
  select
    purchased_at, userid, itemid
  from 
    user_purchased
  -- where [optional filtering]
  --  purchased_at >= xxx -- divide training/test data by time
  CLUSTER BY
    user_id -- Note CLUSTER BY is mandatory when using each_top_k
) t;
```

## Step 2: Recommend top-k items based on users' recently purchased items

In order to generate a list of recommended items, you can use either cooccurrence count or similarity as a relevance score.

### Cooccurrence-based

```sql
WITH topk as (
  select
    each_top_k(
       5, userid, cnt,
       userid, other
    ) as (rank, cnt, userid, rec_item)
  from (
    select 
      t1.userid, t2.other, max(t2.cnt) as cnt
    from
      recently_purchased_items t1
      JOIN cooccurrence t2 ON (t1.itemid = t2.itemid)
    where
      t1.itemid != t2.other -- do not include items that user already purchased
      AND NOT EXISTS (
        SELECT a.itemid FROM user_purchased a
        WHERE a.userid = t1.userid AND a.itemid = t2.other
--        AND a.purchased_count <= 1 -- optional
      )
    group by
      t1.userid, t2.other
    CLUSTER BY
      userid -- top-k grouping by userid
  ) t1
)
INSERT OVERWRITE TABLE item_recommendation
select
  userid,
  map_values(to_ordered_map(rank, rec_item)) as rec_items
from
  topk
group by
  userid
;
```

### Similarity-based

```sql
WITH topk as (
  select
    each_top_k(
       5, userid, similarity,
       userid, other
    ) as (rank, similarity, userid, rec_item)
  from (
    select
      t1.userid, t2.other, max(t2.similarity) as similarity
    from
      recently_purchased_items t1
      JOIN item_similarity t2 ON (t1.itemid = t2.itemid)
    where
      t1.itemid != t2.other -- do not include items that user already purchased
      AND NOT EXISTS (
        SELECT a.itemid FROM user_purchased a
        WHERE a.userid = t1.userid AND a.itemid = t2.other
--        AND a.purchased_count <= 1 -- optional
      )
    group by
      t1.userid, t2.other
    CLUSTER BY
      userid -- top-k grouping by userid
  ) t1
)
INSERT OVERWRITE TABLE item_recommendation
select
  userid,
  map_values(to_ordered_map(rank, rec_item)) as rec_items
from
  topk
group by
  userid
;
```

# Efficient similarity computation

Since naive similarity computation takes `O(n^2)` computational complexity, utilizing a certain approximation scheme is practically important to improve efficiency and feasibility. In particular, Hivemall enables you to use one of two sophisticated approximation schemes, [MinHash](##minhash-compute-pseudo-jaccard-similarity) and [DIMSUM](#dimsum-approximated-all-pairs-cosine-similarity-computation).

## MinHash: Compute "pseudo" Jaccard similarity

Refer [this article](https://en.wikipedia.org/wiki/MinHash#Jaccard_similarity_and_minimum_hash_values
) to get details about MinHash and Jarccard similarity. [This blog article](https://blog.treasuredata.com/blog/2016/02/16/minhash-in-hivemall/) also explains about Hivemall's minhash.

```sql
INSERT OVERWRITE TABLE minhash -- results in 30x records of item_features
select  
  -- assign 30 minhash values for each item
  minhash(itemid, feature_vector, "-n 30") as (clusterid, itemid) -- '-n' would be 10~100
from
  item_features
;

WITH t1 as (
  select
    l.itemid,
    r.itemid as other,
    count(1) / 30 as similarity -- Pseudo jaccard similarity '-n 30'
  from
    minhash l 
    JOIN minhash r 
      ON (l.clusterid = r.clusterid)
  where 
    l.itemid != r.itemid
  group by
    l.itemid, r.itemid
  having
    count(1) >= 3 -- [optional] filtering equals to (count(1)/30) >= 0.1
),
top100 as (
  select
    each_top_k(100, itemid, similarity, itemid, other)
      as (rank, similarity, itemid, other)
  from (
    select * from t1 
    -- where similarity >= 0.1 -- Optional filtering. Can be ignored.
    CLUSTER BY itemid 
  ) t2
)
INSERT OVERWRITE TABLE jaccard_similarity
select
  itemid, other, similarity
from
  top100
;
```

> #### Note
> There might be no similar item for certain items.

### Compute approximated cosine similarity by using the MinHash-based Jaccard similarity

Once the MinHash-based approach found rough `top-N` similar items, you can efficiently find `top-k` similar items in terms of cosine similarity, where `k << N` (e.g., k=10 and N=100).

```sql
WITH similarity as (
  select
    o.itemid,
    o.other,
    cosine_similarity(t1.feature_vector, t2.feature_vector) as similarity
  from
    jaccard_similarity o
    JOIN item_features t1 ON (o.itemid = t1.itemid)
    JOIN item_features t2 ON (o.other = t2.itemid)
),
topk as (
  select
    each_top_k( -- get top-10 items based on similarity score
      10, itemid, similarity,
      itemid, other -- output items
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    where similarity > 0 -- similarity > 0.01
    CLUSTER BY itemid
  ) t
)
INSERT OVERWRITE TABLE cosine_similarity
select 
  itemid, other, similarity
from 
  topk;
```

## DIMSUM: Approximated all-pairs "Cosine" similarity computation

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

DIMSUM is a technique to efficiently and approximately compute [Cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity) for all-pairs of items. You can refer to [an article in Twitter's Engineering blog](https://blog.twitter.com/engineering/en_us/a/2014/all-pairs-similarity-via-dimsum.html) to learn how DIMSUM reduces running time.

Here, let us begin with the `user_purchased` table. `item_similarity` table can be obtained as follows:

```sql
create table item_similarity as
with item_magnitude as ( -- compute magnitude of each item vector
  select
    to_map(j, mag) as mags
  from (
    select 
      itemid as j,
      l2_norm(ln(purchase_count+1)) as mag -- use scaled value
    from 
      user_purchased
    group by
      itemid
  ) t0
),
item_features as (
  select
    userid as i,
    collect_list(
      feature(itemid, ln(purchase_count+1)) -- use scaled value
    ) as feature_vector
  from
    user_purchased
  group by
    userid
),
partial_result as ( -- launch DIMSUM in a MapReduce fashion
  select
    dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.5')
      as (itemid, other, s)
  from
    item_features f
  left outer join item_magnitude m
),
similarity as ( -- reduce (i.e., sum up) mappers' partial results
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
    each_top_k( -- get top-10 items based on similarity score
      10, itemid, similarity,
      itemid, other -- output items
    ) as (rank, similarity, itemid, other)
  from (
    select * from similarity
    CLUSTER BY itemid
  ) t
)
-- insert overwrite table item_similarity
select 
  itemid, other, similarity
from 
  topk
;
```

Ultimately, using `item_similarity` for [item-based recommendation](#item-based-recommendation) is straightforward in a similar way to what we explained above.

In the above query, an important part is obviously `dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.5')`. An option `-threshold` is a real value in `[0, 1)` range, and intuitively it illustrates *"similarities above this threshold are approximated by the DIMSUM algorithm"*.

### Create `item_similarity` from Upper Triangular Matrix

Thanks to the symmetric property of similarity matrix, DIMSUM enables you to utilize space-efficient Upper-Triangular-Matrix-style output by just adding an option `-disable_symmetric_output`:

```sql
create table item_similarity as
with item_magnitude as (
  ...
),
partial_result as (
  select
    dimsum_mapper(f.feature_vector, m.mags, '-threshold 0.5 -disable_symmetric_output')
      as (itemid, other, s)
  from
    item_features f
  left outer join item_magnitude m
),
similarity_upper_triangular as ( -- if similarity of (i1, i2) pair is in this table, (i2, i1)'s similarity is omitted
  select
    itemid, 
    other,
    sum(s) as similarity
  from 
    partial_result
  group by
    itemid, other
),
similarity as ( -- copy (i1, i2)'s similarity as (i2, i1)'s one
  select itemid, other, similarity from similarity_upper_triangular
  union all
  select other as itemid, itemid as other, similarity from similarity_upper_triangular
),
topk as (
  ...
```