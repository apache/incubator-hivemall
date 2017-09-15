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

# Ranking Problems

Practical machine learning applications such as information retrieval and recommendation internally solve ranking problem which generates and returns a ranked list of items. Hivemall provides a way to tackle the problems as follows:

- [Efficient top-k query processing](../misc/topk.html)
- [Recommendation based on item-based collaborative filtering](../recommend/item_based_cf.html)

This page focuses on evaluation of the results from such ranking problems.

> #### Caution
> In order to obtain ranked list of items, this page introduces queries using `to_ordered_map()` such as `map_values(to_ordered_map(score, itemid, true))`. However, this kind of usage has a potential issue that multiple `itemid`-s (i.e., values) which have the exactly same `score` (i.e., key) will be aggregated to single arbitrary `itemid`, because `to_ordered_map()` creates a key-value map which uses duplicated `score` as key.
>
> Hence, if map key could duplicate on more then one map values, we recommend you to use `to_ordered_list(value, key, '-reverse')` instead of `map_values(to_ordered_map(key, value, true))`. The alternative approach is available from Hivemall v0.5-rc.1 or later.

# Binary Response Measures

In a context of ranking problem, **binary response** means that binary labels are assigned to items, and positive items are considered as *truth* observations.

In a `dummy_truth` table, we assume that there are three users (`userid = 1, 2, 3`) who have exactly same three truth ranked items (`itemid = 1, 2, 4`) chosen from existing six items:

| userid | itemid |
| :-: | :-: |
| 1 | 1 |
| 1 | 2 |
| 1 | 4 |
| 2 | 1 |
| 2 | 2 |
| 2 | 4 |
| 3 | 1 |
| 3 | 2 |
| 3 | 4 |

Additionally, here is a `dummy_rec` table we obtained as a result of prediction:

| userid | itemid | score |
| :-: | :-: | :-: |
| 1 | 1 | 10.0 |
| 1 | 3 | 8.0 |
| 1 | 2 | 6.0 |
| 1 | 6 | 2.0 |
| 2 | 1 | 10.0 |
| 2 | 3 | 8.0 |
| 2 | 2 | 6.0 |
| 2 | 6 | 2.0 |
| 3 | 1 | 10.0 |
| 3 | 3 | 8.0 |
| 3 | 2 | 6.0 |
| 3 | 6 | 2.0 |

How can we compare `dummy_rec` with `dummy_truth` to figure out the accuracy of `dummy_rec`?

To be more precise, in case we built a recommender system, let a target user $$u \in \mathcal{U}$$, set of all items $$\mathcal{I}$$, ordered set of top-k recommended items $$I_k(u) \subset \mathcal{I}$$, and set of truth items $$\mathcal{I}^+_u$$. Hence, when we launch top-2 recommendation for the above tables, $$\mathcal{U} = \{1, 2, 3\}$$, $$\mathcal{I} = \{1, 2, 3, 4, 5, 6\}$$ and $$I_2(u) = \{1, 3\}$$ which consists of two highest-scored items, and $$\mathcal{I}^+_u = \{1, 2, 4\}$$.

Evaluation of the ordered sets can be done by the following query:

```sql
with truth as (
  select userid, collect_set(itemid) as truth
  from dummy_truth
  group by userid
),
rec as (
  select
    userid,
    -- map_values(to_ordered_map(score, itemid, true)) as rec,
    to_ordered_list(itemid, score, '-reverse') as rec,
    cast(count(itemid) as int) as max_k
  from dummy_rec
  group by userid
)
select
  -- rec = [1,3,2,6], truth = [1,2,4] for each user
	
  -- Recall@k
  recall_at(t1.rec, t2.truth, t1.max_k) as recall,
  recall_at(t1.rec, t2.truth, 2) as recall_at_2,

  -- Precision@k
  precision_at(t1.rec, t2.truth, t1.max_k) as precision,
  precision_at(t1.rec, t2.truth, 2) as precision_at_2,

  -- MAP
  average_precision(t1.rec, t2.truth, t1.max_k) as average_precision,
  average_precision(t1.rec, t2.truth, 2) as average_precision_at_2,

  -- AUC
  auc(t1.rec, t2.truth, t1.max_k) as auc,
  auc(t1.rec, t2.truth, 2) as auc_at_2,

  -- MRR
  mrr(t1.rec, t2.truth, t1.max_k) as mrr,
  mrr(t1.rec, t2.truth, 2) as mrr_at_2,

  -- NDCG
  ndcg(t1.rec, t2.truth, t1.max_k) as ndcg,
  ndcg(t1.rec, t2.truth, 2) as ndcg_at_2
from rec t1
join truth t2 on (t1.userid = t2.userid)
;
```

We have six different measures, and outputs will be:

| Ranking measure | top-4 (max_k) | top-2 |
| :-: | :-- | :-- |
| Recall | 0.6666666666666666 | 0.3333333333333333 |
| Precision | 0.5 | 0.5 |
| MAP | 0.5555555555555555 | 0.3333333333333333 |
| AUC | 0.75 | 1.0 |
| MRR | 1.0 | 1.0 |
| NDCG | 0.7039180890341349 | 0.6131471927654585 |

Here, we introduce the six measures for evaluation of ranked list of items. Importantly, each metric has a different concept behind formulation, and the accuracy measured by the metrics shows different values even for the exactly same input as demonstrated above. Thus, evaluation using multiple ranking measures is more convincing, and it should be easy in Hivemall.

> #### Caution
> Before Hivemall v0.5-rc.1, `recall_at()` and `precision_at()` are respectively registered as `recall()` and `precision()`. However, since `precision` is a reserved keyword from Hive v2.2.0, [we renamed the function names](https://issues.apache.org/jira/browse/HIVEMALL-140). If you are still using `recall()` and/or `precision()`, we strongly recommend you to use the latest version of Hivemall and replace them with the newer function names.

## Recall-At-k

**Recall-at-k (Recall@k)** indicates coverage of truth samples as a result of top-k recommendation. The value is computed by the following equation:
$$
\mathrm{Recall@}k = \frac{|\mathcal{I}^+_u \cap I_k(u)|}{|\mathcal{I}^+_u|}.
$$
Here, $$|\mathcal{I}^+_u \cap I_k(u)|$$ is the number of true positives. If $$I_2(u) = \{1, 3\}$$ and $$\mathcal{I}^+_u = \{1, 2, 4\}$$, $$\mathrm{Recall@}2 = 1 / 3 \approx 0.333$$.

## Precision-At-k

Unlike Recall@k, **Precision-at-k (Precision@k)** evaluates correctness of a top-k recommendation list $$I_k(u)$$ according to the portion of true positives in the list as:
$$
\mathrm{Precision@}k = \frac{|\mathcal{I}^+_u \cap I_k(u)|}{|I_k(u)|}.
$$
In other words, Precision@k means how much the recommendation list covers true pairs. Here, $$\mathrm{Precision@}2 = 1 / 2 = 0.5$$ where $$I_2(u) = \{1, 3\}$$ and $$\mathcal{I}^+_u = \{1, 2, 4\}$$.

## Mean Average Precision (MAP)

While the original Precision@k provides a score for a fixed-length recommendation list $$I_k(u)$$, **mean average precision (MAP)** computes an average of the scores over all recommendation sizes from 1 to $$|\mathcal{I}|$$. MAP is formulated with an indicator function for $$i_n$$ (the $$n$$-th item of $$I(u)$$), as:
$$
\mathrm{MAP} = \frac{1}{|\mathcal{I}^+_u|} \sum_{n = 1}^{|\mathcal{I}|} \mathrm{Precision@}n \cdot  [ i_n \in \mathcal{I}^+_u ].
$$

It should be noticed that, MAP is not a simple mean of sum of Precision@1, Precision@2, ..., Precision@$$|\mathcal{I}|$$, and higher-ranked true positives lead better MAP. To give an example,
$$
\mathrm{MAP}(\mathcal{I}^+_u, \{1, 3, 2, 6, 4 , 5\}) = \frac{\frac{1}{1} + \frac{2}{3} + \frac{3}{5}}{3} \approx \mathbf{0.756},
$$
where $$\mathcal{I}^+_u = \{1, 2, 4\}$$, while 
$$
\mathrm{MAP}(\mathcal{I}^+_u, \{1, 3, 2, 4, 6, 5\}) = \frac{\frac{1}{1} + \frac{2}{3} + \frac{3}{4}}{3} \approx \mathbf{0.806}.
$$

## Area Under the ROC Curve (AUC)

ROC curve and **area under the ROC curve (AUC)** are generally used in evaluation of the classification problems [as we described before](auc.html). However, these concepts can also be interpreted in a context of ranking problem. 

Basically, the AUC metric for ranking considers all possible pairs of truth and other items which are respectively denoted by $$i^+ \in \mathcal{I}^+_u$$ and $$i^- \in \mathcal{I}^-_u$$, and it expects that the *best* recommender completely ranks $$i^+$$ higher than $$i^-$$. A score is finally computed as portion of the correct ordered $$(i^+, i^-)$$ pairs in the all possible combinations determined by $$|\mathcal{I}^+_u| \times |\mathcal{I}^-_u|$$ in set notation. 

## Mean Reciprocal Rank (MRR)

If we are only interested in the first true positive, **mean reciprocal rank (MRR)** could be a reasonable choice to quantitatively assess the recommendation lists. For $$n_{\mathrm{tp}} \in \left[ 1, |\mathcal{I}| \right]$$, a position of the first true positive in $$I(u)$$, MRR simply returns its inverse:
$$
  \mathrm{MRR} = \frac{1}{n_{\mathrm{tp}}}.
$$
MRR can be zero if and only if $$\mathcal{I}^+_u$$ is empty.

In our dummy tables depicted above, the first true positive is placed at the first place in the ranked list of items. Hence, $$\mathrm{MRR} = 1/1 = 1$$, the best result on this metric.

## Normalized Discounted Cumulative Gain (NDCG)

**Normalized discounted cumulative gain (NDCG)** computes a score for $$I(u)$$ which places emphasis on higher-ranked true positives. In addition to being a more well-formulated measure, the difference between NDCG and MPR is that NDCG allows us to specify an expected ranking within $$\mathcal{I}^+_u$$; that is, the metric can incorporate $$\mathrm{rel}_n$$, a relevance score which suggests how likely the $$n$$-th sample is to be ranked at the top of a recommendation list, and it directly corresponds to an expected ranking of the truth samples.

As a result of top-k recommendation, NDCG is computed by:
$$
\mathrm{NDCG}_k = \frac{\mathrm{DCG}_k}{\mathrm{IDCG}_k} = \frac{\sum_{n=1}^{|\mathcal{I}|} D_k(n) \left[i_n \in \mathcal{I}^+_u\right]}{\sum_{n=1}^{|\mathcal{I}|} D_k(n)},
$$
where
$$
D_k(n) = \left\{
\begin{array}{ll}
  (2^{\mathrm{rel}_n} - 1) / \log_2(n + 1) & (1 \leq n \leq k) \\
  0 & (n > k)
\end{array}
\right.
.
$$
Here, $$\mathrm{DCG}_k$$ indicates how well $$I(u)$$ fits to the truth permutation, and $$\mathrm{IDCG}_k$$ is the best $$\mathrm{DCG}_k$$ that $$I(u)$$ exactly matches to $$\mathcal{I}^+_u$$. 

Now, we only consider binary responses, so relevance score is binary as:
$$
\mathrm{rel}_n = \left\{
\begin{array}{ll}
  1 & (i_n \in \mathcal{I}^+_u) \\
  0 & (\mathrm{otherwise})
\end{array}
\right.
.
$$

Since our recommender launched top-2 recommendation on top of this chapter, $$\mathrm{IDCG}_2 = 1/\log_2 2 + 1/\log_2 3 \approx 1.631$$. Meanwhile, only the first sample in $$I_2(u)$$ is true positive, so $$\mathrm{DCG}_2 = 1/\log_2 2 = 1$$. Hence, $$\mathrm{NDCG}_2 = \mathrm{DCG}_2 / \mathrm{IDCG}_2 \approx 0.613$$.

# Graded Response Measures

While the binary response setting simply considers positive-only ranked list of items, **graded response** additionally handles expected rankings (scores) of the items. Hivemall's NDCG implementation with non-binary relevance score $$\mathrm{rel}_n$$ enables you to evaluate based on the graded responses.

Unlike separated `dummy_truth` and `dummy_rec` table in the binary setting, we assume the following single table named `dummy_recrel` which contains item-$$\mathrm{rel}_n$$ pairs:

| userid | itemid | score<br/>(predicted) | relscore<br/>(expected) |
| :-: | :-: | :-: | :-: |
| 1 | 1 | 10.0 | 5.0 |
| 1 | 3 | 8.0 | 2.0 |
| 1 | 2 | 6.0 | 4.0 |
| 1 | 6 | 2.0 | 1.0 |
| 1 | 4 | 1.0 | 3.0 |
| 2 | 1 | 10.0 | 5.0 |
| 2 | 3 | 8.0 | 2.0 |
| 2 | 2 | 6.0 | 4.0 |
| 2 | 6 | 2.0 | 1.0 |
| 2 | 4 | 1.0 | 3.0 |
| 3 | 1 | 10.0 | 5.0 |
| 3 | 3 | 8.0 | 2.0 |
| 3 | 2 | 6.0 | 4.0 |
| 3 | 6 | 2.0 | 1.0 |
| 3 | 4 | 1.0 | 3.0 |

The function `ndcg()` can take non-binary `truth` values as the second argument: 

```sql
with truth as (
  select
    userid,
    to_ordered_list(relscore, '-reverse') as truth
  from
    dummy_recrel
  group by
    userid
),
rec as (
  select
    userid,
    to_ordered_list(struct(relscore, itemid), score, "-reverse") as rec,
    count(itemid) as max_k
  from
    dummy_recrel
  group by
    userid
)
select 
  -- top-2 recommendation
  ndcg(t1.rec, t2.truth, 2), -- => 0.8128912838590544
  -- top-3 recommendation
  ndcg(t1.rec, t2.truth, 3)  -- => 0.9187707805346093
from
  rec t1
  join truth t2 on (t1.userid = t2.userid)
;
```
