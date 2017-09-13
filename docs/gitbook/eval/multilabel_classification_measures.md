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


Multi-label classification problem is a task to predict labels given two or more categories.

Each sample $$i$$ has $$l_i$$ labels, where $$L$$ is a set of unique labels in the dataset, and $$0 \leq  l_i \leq |L|$$.
This page focuses on evaluation of such multi-label classification problems.

# Example

This page introduces toy example dataset for explanation.

## Data

The following table shows examples of multi-label classification's prediction.

Suppose that animal names represent tags of blog posts and the given task is to predict tags for blog posts.
The left column shows the ground truth labels and the right column shows predicted labels by a multi-label classifier.

| truth labels| predicted labels |
|:---:|:---:|
| cat, bird | cat, dog|
| cat, dog | cat, bird|
| cat | (*no truth label*)|
| bird | bird |
| bird, cat | bird, cat|
| cat, dog | cat, dog, bird |
| dog, bird | dog |


# Evaluation metrics for multi-label classification

Hivemall provides micro F1-score and micro F-measure.

Define $$L$$ is the set of the tag of blog posts, and $$l_i$$ is a tag set of $$i$$-th document.
In the same manner, $$p_i$$ is a predicted tag set of $$i$$-th document.

## Micro F1-score

F1-score is the harmonic mean of recall and precision.

The value is computed by the following equation:

$$
\mathrm{F}_1 = 2 \frac
{\sum_i |l_i \cap p_i |}
{ 2* \sum_i |l_i \cap p_i | + \sum_i |l_i - p_i| + \sum_i |p_i - l_i| }
$$

> #### Caution
> Hivemall also provides `f1score` function, but it is old function to obtain F1-score. The value of `f1score` is based on set operation. So, we recommend to use `fmeasure` function to get F1-score based on this article.

The following query shows the example to obtain F1-score.

```sql
WITH data as (
  select array("cat", "bird") as actual, array("cat", "dog")         as predicted
union all
  select array("cat", "dog")  as actual, array("cat", "bird")        as predicted
union all
  select array("cat")         as actual, array()                     as predicted
union all
  select array("bird")        as actual, array("bird")               as predicted
union all
  select array("bird", "cat") as actual, array("bird", "cat")        as predicted
union all
  select array("cat", "dog")  as actual, array("cat", "dog", "bird") as predicted
union all
  select array("dog", "bird") as actual, array("dog")                as predicted
)
select
  fmeasure(actual, predicted)
from data
;
```

> 0.6956521739130435

## Micro F-measure


F-measure is generalized F1-score and the weighted harmonic mean of recall and precision.

The value is computed by the following equation:
$$
\mathrm{F}_{\beta} = (1+\beta^2) \frac
{\sum_i |l_i \cap p_i |}
{ \beta^2 (\sum_i |l_i \cap p_i | + \sum_i |l_i - p_i|) + \sum_i |l_i \cap p_i | + \sum_i |p_i - l_i|}
$$

$$\beta$$ is the parameter to determine the weight of precision.
So, F1-score is the special case of F-measure given $$\beta=1$$.

If $$\beta$$ is larger positive value than `1.0`, F-measure reaches micro recall.
On the other hand,
if $$\beta$$ is smaller positive value than `1.0`, F-measure reaches micro precision.

If $$\beta$$ is omitted, hivemall calculates F-measure with $$\beta=1$$ (: equivalent to F1-score).

The following query shows the example to obtain F-measure with $$\beta=2$$.

```sql
WITH data as (
  select array("cat", "bird") as actual, array("cat", "dog")         as predicted
union all
  select array("cat", "dog")  as actual, array("cat", "bird")        as predicted
union all
  select array("cat")         as actual, array()                     as predicted
union all
  select array("bird")        as actual, array("bird")               as predicted
union all
  select array("bird", "cat") as actual, array("bird", "cat")        as predicted
union all
  select array("cat", "dog")  as actual, array("cat", "dog", "bird") as predicted
union all
  select array("dog", "bird") as actual, array("dog")                as predicted
)
select
  fmeasure(actual, predicted, '-beta 2.')
from data
;
```

> 0.6779661016949152
