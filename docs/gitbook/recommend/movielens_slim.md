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


SLIM needs top-k most similar items for each item to calculate approximately. 
Here, we particularly focus on [DIMSUM](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation), an efficient and approximated similarity computation scheme, and try to make recommendation from the MovieLens data.


<!-- toc -->

# Compute movie-movie similarity

[As we explained in the general introduction of item-based CF](item_based_cf.html#dimsum-approximated-all-pairs-cosine-similarity-computation.md), following query finds top-$$k$$ nearest-neighborhood movies for each movie:

```sql
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
We can adjust trade-off by varying `k`.
Large `k` is good approximation for raw user-item maxtrix, but training time may increase.

> #### Caution
> To run query above, we may need to run two statement before query above. 
```sql
set hive.strict.checks.cartesian.product=false;
set hive.mapred.mode=nonstrict;
```

# Training

Next, we train SLIM on `train` data.
Here, We store SLIM's weights into table for the recommendation.


# Prediction

Next, we predict rating for unforeseen user-movie pairs based on the top-$$k$$ similarities:

```sql
```
