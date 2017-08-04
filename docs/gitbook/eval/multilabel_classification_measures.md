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

<!-- toc -->

# Multi-label classification


Multi-label classification problem is predicting the labels given categorized dataset.
Each sample $$i$$ has $$l_i$$ labels ($$0 \leq  l_i \leq |L| $$  )
, where $$L$$ is the number of unique labels in the geven dataset.

This page focuses on evaluation of the results from such Multi-label classification problems.

# Examples

For the metrics explanation, this page introduces toy example data and two metrics.

## Data

The following table shows the sample of Multi-label classification's prediction.
Animal names represent the tags of blog post.
Left column includes supervised labels,
Right column includes are predicted labels by a Multi-label classifier.

| truth labels| predicted labels |
|:---:|:---:|
|cat, dog | cat, bird |
| cat, bird | cat, dog |
| | cat |
| bird | bird |
| bird, cat | bird, cat |
| cat, dog, bird | cat, dog |
| dog | dog, bird|


# Evaluation metrics for multi-label classification

Hivemall provises micro F1-score and micro F-measure.

Given $$N$$ blog posts, we uses 

Define $$L$$ is the set of the tag of blog posts, and 
$$l_i$$ is a tag set of $$i$$th document.
In the same manner,
$$p_i$$ is a predicted tag set of $$i$$th document.



## Micro F1-score


F1-score is the harmonic mean of recall and precision.

The value is computed by the following equation:

$$
\mathrm{F}_1 = 2 \frac
{\sum_i |l_i \cap p_i |}
{ 2* \sum_i |l_i \cap p_i | + \sum_i |l_i - p_i | + \sum_i |p_i - l_i | }
$$

The Following query shows the example to obtain F1-score.

```sql
WITH data as (
  select array("cat", "dog")         as actual, array("cat", "bird") as predicted
union all
  select array("cat", "bird")        as actual, array("cat", "dog")  as predicted
union all
  select array()                     as actual, array("cat")         as predicted
union all
  select array("bird")               as actual, array("bird")        as predicted
union all
  select array("bird", "cat")        as actual, array("bird", "cat") as predicted
union all
  select array("cat", "dog", "bird") as actual, array("cat", "dog")  as predicted
union all
  select array("dog")                as actual, array("dog", "bird") as predicted
)
select
  f1score(actual, predicted)
from data
;

--- 0.6956521739130435;
```


## Micro F-measure

F-measure is generalized F1-score and the weighted harmonic mean of recall and precision.

$$\beta$$ is the parameter to determine the weight of precision.
So, F1-score is the special case of F-measure given $$\beta=1$$.

If $$\beta$$ is larger positive value than `1.0`, F-measure reaches to micro recall.
On the other hand,
if $$\beta$$ is smaller positive value than `1.0`, F-measure reaches to micro precision.

The following query shows the example to obtain F-measure with $$\beta=2$$.

$$
\mathrm{F}_{\beta} = (1+\beta^2) \frac
{\sum_i |l_i \cap p_i |}
{ \beta^2 (\sum_i |l_i \cap p_i | + \sum_i |p_i - l_i |) + \sum_i |l_i \cap p_i | + \sum_i |l_i - p_i |}
$$


```sql
WITH data as (
  select array("cat", "dog")         as actual, array("cat", "bird") as predicted
union all
  select array("cat", "bird")        as actual, array("cat", "dog")  as predicted
union all
  select array()                     as actual, array("cat")         as predicted
union all
  select array("bird")               as actual, array("bird")        as predicted
union all
  select array("bird", "cat")        as actual, array("bird", "cat") as predicted
union all
  select array("cat", "dog", "bird") as actual, array("cat", "dog")  as predicted
union all
  select array("dog")                as actual, array("dog", "bird") as predicted
)
select
  fmeasure(actual, predicted, 2)
from data
;

--- 0.6956521739130435;
```
