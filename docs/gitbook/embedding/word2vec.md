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
Word Embedding is a powerful tool for many tasks,
e.g. finding similar words,
features for supervised machine learning task and word analogy,
such as `king - man + woman =~ queen`.
In word embedding,
each word represents a low dimension and dense vector representation.
**Skip-gram** and **Continuous Bag-of-words** (a.k.a word2vec) are the most popular algorithms to obtain good word embeddings.

The papers introduce the method are as follows:

- T. Mikolov, et al., [Distributed Representations of Words and Phrases and Their Compositionality
](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). NIPS, 2013.
- T. Mikolov, et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). ICLR, 2013.

Hivemall provides two type algorithms: Skip-gram and Continuous Bag-of-words (CBoW) with negative sampling.
Hivemall enables you to train your sequence data such as,
but not limited to, documents based on word2vec.
This article gives usage instructions of the feature.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.? or later.

# Prepare document data

Assume that you already have a table `docs` which contains many documents as string format:

```sql
select * FROM docs;
```

| docId | doc |
|:----: |:----|
|   0   | "Alice was beginning to get very tired of sitting by her sister on the bank ..." |
|  ...  | ... |

First, each document is split into words by tokenize function like a `tokenize`.

```sql
drop table docs_words;
create table docs_words as
  select
    docid,
    tokenize(doc, true) as words
  FROM
    docs
;
```

| docId | doc |
|:----: |:----|
|   0   | "alice", "was", "beginning", "to", "get", "very", "tired", "of", "sitting", "by", "her", "sister", "on", "the", "bank", ... |
|  ...  | ... |

Then, you count all word frequency and remove low frequency words from vocabulary.
Removing low frequency words is optinal, but it is better for getting word vector fastly.

```sql
set hivevar:mincount=5;

drop table freq;
create table freq as
select
  word,
  freq
from (
    select
      word,
      COUNT(*) as freq
    from
      docs_words
    LATERAL VIEW explode(words) lTable as word
    group by
      word
) t
where freq >= ${mincount}
;
```

`numTrainWords` is set the number of words in corpus exclude low frequency words.

```sql
select sum(freq) FROM freq;

-- set variable query above result
set hivevar:numTrainWords=750105;
```

# Create sub-sampling table

Sub-sampling table is stored a not deleted probability per word.
During word2vec training,
sub-sampled words are ignored.
It works to train fastly and to consider the imbalance the rare words and frequent words by reducing frequent words.
If you want to know detail, 
please check Eq.5 in the [original paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

```sql
set hivevar:sample=1e-4;

drop table subsampling_table;
create table subsampling_table as
  select * FROM (
    select
        word,
        sqrt(${sample}/(freq/${numTrainWords})) + ${sample}/(freq/${numTrainWords}) as p
    from freq
) t
;
```

# Delete low frequency words and high frequency words from `docs_words`

To reduce useless words from corpus,
low frequency words and high frequency words are deleted.
And, to avoid loading on memory, a long document is split into some sub-documents.

```sql
set hivevar:maxlength=1500;
SET hivevar:seed=31;

drop table train_docs;
create table train_docs as
  with docs_exploded as (
    select
      docid,
      word,
      pos % ${maxlength} as pos,
      pos div ${maxlength} as splitid,
      rand(${seed}) as rnd
    from
      docs_words LATERAL VIEW posexplode(words) t as pos, word
  )
select
    docid,
    to_ordered_list(l.word, pos) as words
from
  docs_exploded l
join freq r on (l.word = r.word)
join subsampling_table r2 on (l.word = r2.word)
where r2.p > l.rnd
group by
  docid, splitid
;
```

# Create negative sampling table

Negative sampling is an approximate function of [softmax function](https://en.wikipedia.org/wiki/Softmax_function).
Here, `negative_table` is used to store word sampling probability for negative sampling.
`noisePower` is a hyperparameter of noise distribution for negative sampling.
During word2vec training,
words sampled this distribution are used for negative examples.

```sql
set hivevar:noisePower=3/4;

drop table negative_table;
create table negative_table as
  select
    word,
    pow(freq, ${noisePower}) as negative
  from
    freq
;
```

## Split negative sampling table

To avoid using huge memory space for negative sampling like original implementation and sample fastly from this distribution,
Hivemall uses [Alias method](https://en.wikipedia.org/wiki/Alias_method).

And then, this alias sampler is split into `numSplit` tables in the next query.

```sql
set hivevar:numSplit=8;

drop table split_negative_table;
create table split_negative_table as
with alias_bins as (
select
    alias_table(to_map(word, negative), ${numSplit})
    as (k, word, p, other)
from
    negative_table
)
select
  k,
  collect_list(array(word, p, other)) as negative_table
from
  alias_bins
group by k
;
```

# Train word2vec

## Create feature

Hivemall provides `word2vec_feature` function to prepare the input of word2vec training.
Default feature is skip-gram.
In this case, `word2vec_feature` function returns the records below.

| inWord | posWord | negWords|
|---- |----|----|
| "alice" | "was" | ["queen", "a", ...] |
|  ...  | ... | ... |

By passing `-model cbow` to `word2vec_feature` argument,
Hivemall creates CBoW feature:

| inWords | posWord | negWords|
|---- |----|----|
| ["alice", "beginning", "to" | "was" | ["queen", "a", ...] |
|  ...  | ... | ... |


### Skip-Gram features

```sql
drop table skipgram_features;
create table skipgram_features as
select
  word2vec_feature(k, negative_table, words, "-win 5 -neg 15 -iter 5")
from(
    select
      r.k,
      negative_table,
      words
    from
      train_docs l
    join split_negative_table r on
      l.docid % ${numSplit} = r.k
    CLUSTER BY r.k
) t
;
```

### CBoW features

```sql
drop table cbow_features;
create table cbow_features as
select
  word2vec_feature(k, negative_table, words, "-win 5 -neg 15 -iter 5 -model cbow")
from(
    select
      r.k,
      negative_table,
      words
    from
      train_docs l
    join split_negative_table r on
      l.docid % ${numSplit} = r.k
    CLUSTER BY r.k
) t
;
```

## Train Word2Vec

In the same way,
`train_word2vec` function is enable to train word vector based on both types features by passing `"-model"`.

### Train Skip-Gram

```sql
select COUNT(*) from skipgram_features;
set hivevar:numSamples=2981487;

drop table w2v_skgram;
create table w2v_skgram as
select word, i, avg(wi) as wi
from (
  select
    train_word2vec(
        inword,
        posword,
        negwords,
        ${numSamples}
    )
    from (
      select * from skipgram_features
      CLUSTER by k
    ) t
) t1
group by
    word, i
;

```

### Train CBoW

```sql
select COUNT(*) from cbow_features;
set hivevar:numSamples=499157;
drop table w2v_cbow;
create table w2v_cbow as
select word, i, avg(wi) as wi
from (
  select
    train_word2vec(
        inwords,
        posword,
        negwords,
        ${numSamples},
        "-model cbow"
    )
    from (
      select * from cbow_features
      CLUSTER by k
    ) t
) t1
group by
    word, i
;
```
