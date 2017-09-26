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
feature vector for supervised machine learning task and word analogy,
such as `king - man + woman =~ queen`.
In word embedding,
each word represents a low dimension and dense vector.
**Skip-Gram** and **Continuous Bag-of-words** (CBoW) are the most popular algorithms to obtain good word embeddings (a.k.a word2vec).

The papers introduce the method are as follows:

- T. Mikolov, et al., [Distributed Representations of Words and Phrases and Their Compositionality
](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). NIPS, 2013.
- T. Mikolov, et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). ICLR, 2013.

Hivemall provides two type algorithms: Skip-gram and CBoW with negative sampling.
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
  row_number() over () - 1 as wordid,
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

# Create sub-sampling table

Sub-sampling table is stored a not deleted probability per word.
During word2vec training,
sub-sampled words are ignored.
It works to train fastly and to consider the imbalance the rare words and frequent words by reducing frequent words.
If you want to know detail of Sub-sampling,
please check Eq.5 in the [original paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

```sql
set hivevar:sample=1e-4;

drop table subsampling_table;
create table subsampling_table as
with stats as (
  select
    sum(freq) as numTrainWords
  FROM
    freq
)
select
  l.wordid,
  l.word,
  sqrt(${sample}/(l.freq/r.numTrainWords)) + ${sample}/(l.freq/r.numTrainWords) as p
from
  freq l
cross join
  stats r
;
```


```sql
select * FROM subsampling_table order by p;
```

| word | p |
|:----: |:----:|
| the | 0.04013665 |
| of | 0.052463654 |
| and | 0.06555538 |
| 00 | 0.068162076 |
| in | 0.071441144 |
| 0 | 0.07528994 |
| a | 0.07559573 |
| to | 0.07953133 |
| 0000 | 0.08779001 |
| is | 0.09049763 |
| 000 | 0.11748954 |
|  ...  | ... |

In this case, 
`the` is used only 4% in the documents during training.

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
  l.docid,
  to_ordered_list(r2.wordid, l.pos) as words
  -- to_ordered_list(l.word, l.pos) as words
from
  docs_exploded l
  LEFT SEMI join freq r on (l.word = r.word)
  join subsampling_table r2 on (l.word = r2.word)
where
  r2.p > l.rnd
group by
  l.docid, l.splitid
;
```

# Create negative sampling table

Negative sampling is an approximate function of [softmax function](https://en.wikipedia.org/wiki/Softmax_function).
Here, `negative_table` is used to store word sampling probability for negative sampling.
`noisePower` is a hyperparameter of noise distribution for negative sampling.
During word2vec training,
words sampled this distribution are used for negative examples.

To avoid using huge memory space for negative sampling like original implementation and sample fastly from this distribution,
Hivemall uses [Alias method](https://en.wikipedia.org/wiki/Alias_method).

## String case

```sql
set hivevar:noisePower=3/4;

drop table negative_table;
create table negative_table as
select
  collect_list(array(word, p, other)) as negative_table
from (
  select
    alias_table(to_map(word, negative)) as (word, p, other)
  from
    (
      select
        word,
        pow(freq, ${noisePower}) as negative
      from
        freq
    ) t
) t1
;
```


## Int case

```sql
set hivevar:noisePower=3/4;

drop table negative_table;
create table negative_table as
select
  collect_list(array(wordid, p, other)) as negative_table
from (
  select
    alias_table(to_map(wordid, negative)) as (wordid, p, other)
  from
    (
      select
        wordid,
        pow(freq, ${noisePower}) as negative
      from
        freq
    ) t
) t1
;
```

# Train word2vec

Hivemall provides `train_word2vec` function to prepare the input of word2vec training.
Default model is `"skipgram"`.

### Skip-Gram

```sql
select sum(size(words)) from train_docs;

drop table skipgram;
create table skipgram as
select
  train_word2vec(
    r.negative_table,
    l.words,
    "-n 418953 -win 5 -neg 15 -iter 5 -dim 100 -model skipgram"
  )
from
  train_docs l
  cross join negative_table r
;
```

### CBoW

```sql
drop table cbow;

create table cbow as
select
  train_word2vec(
    r.negative_table,
    l.words,
    "-n 418953 -win 5 -neg 15 -iter 5 -model cbow"
  )
from
  train_docs l
  cross join negative_table r
;
```
