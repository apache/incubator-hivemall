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
feature vectors for supervised machine learning task and word analogy,
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

Assume that you already have `docs` table which contains many documents as string format with unique index:

```sql
select * FROM docs;
```

| docId | doc |
|:----: |:----|
|   0   | "Alice was beginning to get very tired of sitting by her sister on the bank ..." |
|  ...  | ... |

First, each document is split into words by tokenize function like a [`tokenize`](../misc/tokenizer.html).

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

This table shows tokenized document.

| docId | doc |
|:----: |:----|
|   0   | ["alice", "was", "beginning", "to", "get", "very", "tired", "of", "sitting", "by", "her", "sister", "on", "the", "bank", ...] |
|  ...  | ... |

Then, you count frequency up per word and remove low frequency words from the vocabulary.
To remove low frequency words is optional preprocessing, but this process is effective to train word vector fastly.

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

Hivemall's word2vec supports two type words; string and int.
String type tends to use huge memory during training.
On the other hand, int type tends to use less memory.
If you train on small dataset, we recommend using string type,
because memory usage can be ignored and HiveQL is more simple.
If you train on large dataset, we recommend using int type,
because it saves memory during training.

# Create sub-sampling table

Sub-sampling table is stored a sub-sampling probability per word.

The sub-sampling probability of word $$w_i$$ is computed by the following equation:

$$
\begin{aligned}
f(w_i) = \sqrt{\frac{\mathrm{sample}}{freq(w_i)/\sum freq(w)}} + \frac{\mathrm{sample}}{freq(w_i)/\sum freq(w)}
\end{aligned}
$$

During word2vec training,
not sub-sampled words are ignored.
It works to train fastly and to consider the imbalance the rare words and frequent words by reducing frequent words.
The smaller `sample` value set,
the fewer words are used during training.

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

| wordid | word | p |
|:----: | :----: |:----:|
| 48645 | the  | 0.04013665|
| 11245 | of   | 0.052463654|
| 16368 | and  | 0.06555538|
| 61938 | 00   | 0.068162076|
| 19977 | in   | 0.071441144|
| 83599 | 0    | 0.07528994|
| 95017 | a    | 0.07559573|
| 1225  | to   | 0.07953133|
| 37062 | 0000 | 0.08779001|
| 58246 | is   | 0.09049763|
|  ...  | ...  |... |

The first row shows that 4% of `the` are used in the documents during training.

# Delete low frequency words and high frequency words from `docs_words`

To reduce useless words from corpus,
low frequency words and high frequency words are deleted.
And, to avoid loading long document on memory, a  document is split into some sub-documents.

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
  -- to_ordered_list(l.word, l.pos) as words
  to_ordered_list(r2.wordid, l.pos) as words,
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

If you store string word in `train_docs` table,
please replace `to_ordered_list(r2.wordid, l.pos) as words` with  `to_ordered_list(l.word, l.pos) as words`.

# Create negative sampling table

Negative sampling is an approximate function of [softmax function](https://en.wikipedia.org/wiki/Softmax_function).
Here, `negative_table` is used to store word sampling probability for negative sampling.
`z` is a hyperparameter of noise distribution for negative sampling.
During word2vec training,
words sampled from this distribution are used for negative examples.
Noise distribution is the unigram distribution raised to the 3/4rd power.

$$
\begin{aligned}
p(w_i) = \frac{freq(w_i)^{\mathrm{z}}}{\sum freq(w)^{\mathrm{z}}}
\end{aligned}
$$

To avoid using huge memory space for negative sampling like original implementation and remain to sample fastly from this distribution,
Hivemall uses [Alias method](https://en.wikipedia.org/wiki/Alias_method).

This method has proposed in papers below:

- A. J. Walker, New Fast Method for Generating Discrete Random Numbers with Arbitrary Frequency Distributions, in Electronics Letters 10, no. 8, pp. 127-128, 1974.
- A. J. Walker, An Efficient Method for Generating Discrete Random Variables with General Distributions. ACM Transactions on Mathematical Software 3, no. 3, pp. 253-256, 1977.

```sql
set hivevar:z=3/4;

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
        -- wordid as word,
        pow(freq, ${z}) as negative
      from
        freq
    ) t
) t1
;
```

`alias_table` function returns the records like following.

| word | p | other |
|:----: | :----: |:----:|
| leopold | 0.6556492 | 0000 |
| slep | 0.09060383 | leopold |
| valentinian | 0.76077825 | belarusian |
| slew | 0.90569097 | colin |
| lucien | 0.86329675 | overland |
| equitable | 0.7270946 | farms |
| insurers | 0.2367955 | israel |
| lucier | 0.14855136 | supplements |
| lieve | 0.12075222 | separatist |
| skyhawks | 0.14079945 | steamed |
| ... | ... | ... |

To sample negative word from this `negative_table`,

1. Sample record int index `i` from $$[0 \ldots \mathrm{num\_alias\_table\_records}]$$.
2. Sample float value `r` from $$[0.0 \ldots 1.0]$$ .
3. If `r` < `p` of `i` th record, return `word` `i` th record, else return `other` of `i` th record.

Here, to use it in training function of word2vec, 
`alias_table`'s return records are stored into one list in the `negative_table`.

# Train word2vec

Hivemall provides `train_word2vec` function to train word vector by word2vec algorithms.
The default model is `"skipgram"`.

> #### Note
> You must pass `n` argumet to the number of words in training documents: `select sum(size(words)) from train_docs;`.

## Train Skip-Gram

In skip-gram model,
word vectors are trained to predict the nearby words.
For example, given a sentence like a `"alice", "was", "beginning", "to"`,
`"was"` vector is learnt to predict `"alice"` ,`"beginning"` and `"to"`.

```sql
select sum(size(words)) from train_docs;
set hivevar:n=418953; -- previous query return value

drop table skipgram;
create table skipgram as
select
  train_word2vec(
    r.negative_table,
    l.words,
    "-n ${n} -win 5 -neg 15 -iter 5 -dim 100 -model skipgram"
  )
from
  train_docs l
  cross join negative_table r
;
```

When word is treated as int istead of string,
you may need to transform wordid of int to word of string by `join` statement.

```sql
drop table skipgram;

create table skipgram as
select
  r.word, t.i, t.wi
from (
  select
    train_word2vec(
      r.negative_table,
      l.wordsint,
      "-n 418953 -win 5 -neg 15 -iter 5"
    ) as (wordid, i, wi)
  from
    train_docs l
  cross join
    negative_table r
) t
join freq r on (t.wordid = r.wordid)
;
```

## Train CBoW

In CBoW model,
word vectors are trained to be predicted the nearby words.
For example, given a sentence like a `"alice", "was", "beginning", "to"`,
`"alice"` ,`"beginning"` and `"to"` vectors are learnt to predict `"was"` vector.

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

## Usage of `train_word2vec`

You can get usages of `train_word2vec` by giving `-help` option as follows:

```sql
select
  train_word2vec(
    r.negative_table,
    l.words,
    "-help"
  )
from
  train_docs l
cross join
  negative_table r
;
```

```
usage: train_word2vec(array<array<float | string>> negative_table,
       array<int | string> doc [, const string options]) - Returns a
       prediction model [-dim <arg>] [-help] [-iter <arg>] [-lr <arg>]
       [-model <arg>] [-n <arg>] [-neg <arg>] [-win <arg>]
 -dim,--dimension <arg>     The number of vector dimension [default: 100]
 -help                      Show function help
 -iter,--iteration <arg>    The number of iterations [default: 5]
 -lr,--learningRate <arg>   Initial learning rate of SGD. The default
                            value depends on model [default: 0.025
                            (skipgram), 0.05 (cbow)]
 -model,--modelName <arg>   The model name of word2vec: skipgram or cbow
                            [default: skipgram]
 -n,--numTrainWords <arg>   The number of words in the documents. It is
                            used to update learning rate
 -neg,--negative <arg>      The number of negative sampled words per word
                            [default: 5]
 -win,--window <arg>        Context window size [default: 5]
 ```
