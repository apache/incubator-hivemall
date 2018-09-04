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

Topic modeling is a way to analyze massive documents by clustering them into some ***topics***. In particular, **Latent Dirichlet Allocation** (LDA) is one of the most popular topic modeling techniques; papers introduce the method are as follows:

- D. M. Blei, et al. [Latent Dirichlet Allocation](http://www.jmlr.org/papers/v3/blei03a.html). Journal of Machine Learning Research 3, pp. 993-1022, 2003.
- M. D. Hoffman, et al. [Online Learning for Latent Dirichlet Allocation](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation). NIPS 2010.

Hivemall enables you to analyze your data such as, but not limited to, documents based on LDA. This page gives usage instructions of the feature.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

# Prepare document data

Assume that we already have a table `docs` which contains many documents as string format:

| docid | doc  |
|:---:|:---|
| 1  | "Fruits and vegetables are healthy." |
|2 | "I like apples, oranges, and avocados. I do not like the flu or colds." |
| ... | ... |

Hivemall has several functions which are particularly useful for text processing. More specifically, by using `tokenize()` and `is_stopword()`, you can immediately convert the documents to [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)-like format:

```sql
with word_counts as (
  select
    docid,
    feature(word, count(word)) as word_count
  from 
    docs t1 
    LATERAL VIEW explode(tokenize(doc, true)) t2 as word
  where
    not is_stopword(word)
  group by
    docid, word
)
select docid, collect_list(word_count) as features
from word_counts
group by docid
;
```

| docid | features |
|:---:|:---|
|1  | ["fruits:1","healthy:1","vegetables:1"] |
|2  | ["apples:1","avocados:1","colds:1","flu:1","like:2","oranges:1"] |

> #### Note
> It should be noted that, as long as your data can be represented as the feature format, LDA can be applied for arbitrary data as a generic clustering technique.

# Building Topic Models and Finding Topic Words

Each feature vector is input to the `train_lda()` function:

```sql
with word_counts as (
  select
    docid,
    feature(word, count(word)) as word_count
  from docs t1 LATERAL VIEW explode(tokenize(doc, true)) t2 as word
  where
    not is_stopword(word)
  group by
    docid, word
),
input as (
  select docid, collect_list(word_count) as features
  from word_counts
  group by docid
)
select
  train_lda(features, '-topics 2 -iter 20') as (label, word, lambda)
from
  input
;
```

Here, an option `-topics 2` specifies the number of topics we assume in the set of documents.

Notice that `order by docid` ensures building a LDA model precisely in a single node. In case that you like to launch `train_lda` in parallel, following query hopefully returns similar (but might be slightly approximated) result:

```sql
with word_counts as (
  -- same as above
),
input as (
  select docid, collect_list(f) as features
  from word_counts
  group by docid
)
select
  label, word, avg(lambda) as lambda
from (
  select
    train_lda(features, '-topics 2 -iter 20') as (label, word, lambda)
  from 
    input
) t2
group by label, word
-- order by lambda desc -- ordering is optional
;
```

Eventually, a new table `lda_model` is generated as shown below:

|label | word   | lambda |
|:---:|:---:|:---:|
|0     | fruits | 0.33372128|
|0     | vegetables  |    0.33272517|
|0     | healthy | 0.33246377|
|0     | flu   |  2.3617347E-4|
|0     | apples | 2.1898883E-4|
|0     | oranges | 1.8161473E-4|
|0     | like   | 1.7666373E-4|
|0     | avocados  |      1.726186E-4|
|0     | colds  | 1.037139E-4|
|1     | colds  | 0.16622013|
|1     | avocados |       0.16618845|
|1     | oranges | 0.1661859|
|1     | like  |  0.16618414|
|1     | apples |  0.16616651|
|1     | flu   |  0.16615893|
|1     | healthy | 0.0012059759|
|1     | vegetables  |    0.0010818697|
|1     | fruits  | 6.080827E-4|

In the table, `label` indicates a topic index, and `lambda` is a value which represents how each word is likely to characterize a topic. That is, we can say that, in terms of `lambda`, top-N words are the ***topic words*** of a topic.

Obviously, we can observe that topic `0` corresponds to document `1`, and topic `1` represents words in document `2`.

# Predicting Topic Assignments of Documents

Once you have constructed topic models as described before, a function `lda_predict()` allows you to predict topic assignments of documents.

For example, if we consider the `docs` table, the exactly same set of documents as used for training, probability that a document is assigned to a topic can be computed by:

```sql
with test as (
  select
    docid,
    word,
    count(word) as value
  from
    docs t1
    LATERAL VIEW explode(tokenize(doc, true)) t2 as word
  where
    not is_stopword(word)
  group by
    docid, word
)
select
  t.docid,
  lda_predict(t.word, t.value, m.label, m.lambda, '-topics 2') as probabilities
from
  test t
  JOIN lda_model m ON (t.word = m.word)
group by
  t.docid
;
```

| docid | probabilities (sorted by probabilities) | 
|:---:|:---|
|1  | [{"label":0,"probability":0.875},{"label":1,"probability":0.125}]|
|2  | [{"label":1,"probability":0.9375},{"label":0,"probability":0.0625}]|

Importantly, an option `-topics` is expected to be the same value as you set for training.

Since the probabilities are sorted in descending order, a label of the most promising topic is easily obtained as:

```sql
select docid, probabilities[0].label
from topic;
```

| docid | label |
|:---:|:---:|
|  1 | 0 |
| 2 | 1 |

Of course, using the different set of documents for prediction is possible. Predicting topic assignments of newly observed documents should be more realistic scenario.
