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

As described in [our user guide for Latent Dirichlet Allocation (LDA)](lda.md), Hivemall enables you to apply clustering for your data based on a topic modeling technique. While LDA is one of the most popular techniques, there is another approach named **Probabilistic Latent Semantic Analysis** (pLSA). In fact, pLSA is the predecessor of LDA, but it has an advantage in terms of running time.

- T. Hofmann. [Probabilistic Latent Semantic Indexing](http://dl.acm.org/citation.cfm?id=312649). SIGIR 1999, pp. 50-57.
- T. Hofmann. [Probabilistic Latent Semantic Analysis](http://www.iro.umontreal.ca/~nie/IFT6255/Hofmann-UAI99.pdf). UAI 1999, pp. 289-296.

In order to efficiently handle large-scale data, our pLSA implementation is based on the following incremental variant of the original pLSA algorithm:

- H. Wu, et al. [Incremental Probabilistic Latent Semantic Analysis for Automatic Question Recommendation](http://dl.acm.org/citation.cfm?id=1454026). RecSys 2008, pp. 99-106.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

# Usage

Basically, you can use our pLSA function in a similar way to LDA.

In particular, we have two pLSA functions, `train_plsa()` and `plsa_predict()`. These functions can be used almost interchangeably with `train_lda()` and `lda_predict()`. Thus, reading [our user guide for LDA](lda.md) should be helpful before trying pLSA.

In short, for the sample `docs` table we introduced in the LDA tutorial:

| docid | doc  |
|:---:|:---|
| 1  | "Fruits and vegetables are healthy." |
|2 | "I like apples, oranges, and avocados. I do not like the flu or colds." |
| ... | ... |

a pLSA model can be built as follows:

```sql
with word_counts as (
  select
    docid,
    feature(word, count(word)) as f
  from 
    docs t1
	lateral view explode(tokenize(doc, true)) t2 as word
  where
    not is_stopword(word)
  group by
    docid, word
),
input as (
  select docid, collect_list(f) as features
  from word_counts
  group by docid
)
select
  train_plsa(features, '-topics 2 -eps 0.00001 -iter 2048 -alpha 0.01') as (label, word, prob)
from 
  input
;
```

|label |  word  |  prob|
|:---:|:---:|:---:|
|0|       like   | 0.28549945|
|0|       colds  | 0.14294468|
|0|       apples | 0.14291435|
|0|       avocados|        0.1428958|
|0|       flu    | 0.14287639|
|0|       oranges| 0.1428691|
|0|       healthy| 1.2605103E-7|
|0|       fruits | 4.772253E-8|
|0|       vegetables |     1.929087E-8|
|1|       vegetables  |    0.32713377|
|1|       fruits | 0.32713372|
|1|       healthy| 0.3271335|
|1|       like   | 0.006977764|
|1|       oranges| 0.0025642214|
|1|       flu    | 0.002507711|
|1|       avocados|        0.0023572792|
|1|       apples | 0.002213457|
|1|       colds  | 0.001978546|


And prediction can be done as:

```sql
test as (
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
),
topic as (
  select
    t.docid,
    plsa_predict(t.word, t.value, m.label, m.prob, '-topics 2') as probabilities
  from
    test t
    JOIN plsa_model m ON (t.word = m.word)
  group by
    t.docid
)
select 
  docid, 
  probabilities, 
  probabilities[0].label, 
  m.words -- topic each document should be assigned
from
  topic t 
  JOIN (
    select label, collect_list(feature(word, prob)) as words
    from plsa_model
    group by label
  ) m on t.probabilities[0].label = m.label
;
```


|docid  | probabilities |  label |  m.words |
|:---:|:---|:---:|:---|
|1      | [{"label":1,"probability":0.72298235},{"label":0,"probability":0.27701768}]   |  1 |      ["vegetables:0.32713377","fruits:0.32713372","healthy:0.3271335","like:0.006977764","oranges:0.0025642214","flu:0.002507711","avocados:0.0023572792","apples:0.002213457","colds:0.001978546"]|
|2  |     [{"label":0,"probability":0.7052526},{"label":1,"probability":0.2947474}]     |  0     |  ["like:0.28549945","colds:0.14294468","apples:0.14291435","avocados:0.1428958","flu:0.14287639","oranges:0.1428691","healthy:1.2605103E-7","fruits:4.772253E-8","vegetables:1.929087E-8"]|

# Difference with LDA

The main advantage of using pLSA is its efficiency. Since mathematical formulation and optimization logic is much simpler than LDA, using pLSA generally requires much shorter running time.

In terms of accuracy, LDA could be better than pLSA. For example, a word `like` appears twice in the above sample document#2 gets larger probabilities both in topic#1 and #2, even though one document does not contain the word. By contrast, LDA results (i.e., *lambda* values) are more clearly separated as shown in [the LDA page](lda.md). Thus, a pLSA model is likely to be biased.

For the reasons that we mentioned above, we recommend you to first use LDA. After that, if you encountered problems such as slow running time and undesirable clustering results, let you try alternative pLSA approach.

# Setting hyper-parameter `alpha`

For training pLSA, we set a hyper-parameter `alpha` in the above example:

```sql
SELECT train_plsa(feature, '-topics 2 -eps 0.00001 -iter 2048 -alpha 0.01') 
```

This value controls **how much iterative model update is affected by the old results**.

From an algorithmic point of view, training pLSA (and LDA) iteratively repeats certain operations and updates the target value (i.e., probability obtained as a result of `train_plsa()`). This iterative procedure gradually makes the probabilities more accurate. What `alpha` does is to control the degree of the change of probabilities in each step.

Importantly, pLSA is likely to overfit single mini-batch. As a result, $$P(w|z)$$ could be particularly bad values (i.e., $$(w|z) = 0$$), and `train_plsa()` sometimes fails with an exception like:

```
Perplexity would be Infinity. Try different mini-batch size `-s`, larger `-delta` and/or larger `-alpha`.
```

In that case, you need to try different hyper-parameters to avoid overfitting as the exception suggests.

For instance, [20 newsgroups dataset](http://qwone.com/~jason/20Newsgroups/) which consists of 10906 realistic documents empirically requires the following options:

```sql
SELECT train_plsa(features, '-topics 20 -iter 10 -s 128 -delta 0.01 -alpha 512 -eps 0.1')
```

Clearly, `alpha` is much larger than `0.01` which was used for the dummy data above. Let you keep in mind that an appropriate value of `alpha` highly depends on the number of documents and mini-batch size.
