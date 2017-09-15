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

Papers introduce the method are as follows:

- T. Mikolov, et al., [Distributed Representations of Words and Phrases and Their Compositionality
](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). NIPS, 2013.
- T. Mikolov, et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). ICLR, 2013.

Hivemall provides two type algorithms: Skip-gram and Continuous Bag-of-words (CBoW) with negative sampling for large data.
Hivemall enables you to train your sequence data such as,
but not limited to, documents based on word2vec.
This article gives usage instructions of the feature.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.? or later.

# Prepare document data

Assume that we already have a table `docs` which contains many documents as string format:

```sql
select * FROM docs;
```

| docId | doc |
|:----: |:----|
|  0    | "Alice was beginning to get very tired of sitting by her sister on the bank ..." |
| ...   | ... |

Then, we split each document string into words: list of string.

``` sql
drop table docs_words;
create table docs_words as
  select
    docid,
    tokenize(doc, true) as words
  FROM
    docs
;
```

Then, we count all word frequency and remove low frequency words.

``` sql
set hivevar:mincount=15;

drop table freq;
create table freq as
select word, freq from (
    select word, COUNT(*) as freq
    from docs_words
    LATERAL VIEW explode(words) lTable as word
    group by word
) t
where freq >= ${mincount}
;
```

We set the number of training words `numTrainWords` variable.

``` sql
select sum(freq) FROM freq;

-- set variable query above result
set hivevar:numTrainWords=111200449;
```

# Delete low frequency words from `docs_words`

```sql
drop table docs_exploded;
create table docs_exploded as
  select
    docid, word
  from
    docs_words LATERAL VIEW explode(words) t as word
;

drop table train_docs;
create table train_docs as
select
    docid,
    collect_list(l.word) as words
from docs_exploded l
join freq r on (l.word = r.word)
group by docid;
```

# Create discard table

Discard table is stored a delete probability per word whether delete word or not.
During word2vec training,
discarded words are skipped,
but it is not skipped as context words.

```sql
set hivevar:sample=1e-4;

drop table discard_table;
create table discard_table as
  select * FROM (
    select
        word,
        sqrt(${sample}/(freq/${numTrainWords})) + ${sample}/(freq/${numTrainWords}) as discard
    from freq
) t
where discard < 1.
;
```

# Create negative sampling table

Negative table is used to store word sampling probability for negative sampling.
`noisePower` is a hyperparameter of noise distribution for negative sampling.
During word2vec training, negative words are used for negative example.

```sql
set hivevar:noisePower=3/4;

drop table negative_table;
create table negative_table as
  select
    word,
    pow(freq, ${noisePower}) as negative
  from
    freq;
```

## Split negative sampling table

Negative sampling table is stored all valid words and its probabilities.

To avoid using this huge memory splace and sample fastly from this distribution,
Hivemall uses [Alias method](https://en.wikipedia.org/wiki/Alias_method).

And then, this alias sampler is split into N tables for distributed training on hive.

``` sql
set hivevar:numSplit=16;

drop table split_alias_table;
create table split_alias_table as
with alias_bins as (
select
    alias_table(to_map(word, negative), ${numSplit})
    as (k, word, p, other)
from
    negative_table
) select k, collect_list(array(word, p, other)) as alias
from alias_bins
group by k;
```

# Train word2vec

```sql
drop table w2v;
create table w2v as 
with discard_map as (
  select
    to_map(word, discard) as m
  FROM
    discard_table
) select
    word, i, avg(wi)
  FROM (
    select(
      skipgram(
        words,
        k,
        alias,
        m,
        ${numTrainWords},
        "-win 5 -dim 100 -neg 15"
      )
    )
    from (
      select
        words, r.k, alias, m
      from 
        train_docs l
      join split_alias_table r on
        l.docid % ${numSplit} = r.k
      join discard_map
        CLUSTER BY r.k
    ) t
  ) t1
group by word, i
;
```
