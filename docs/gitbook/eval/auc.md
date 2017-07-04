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

# Area Under the ROC Curve

[ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and Area Under the ROC Curve (AUC) are widely-used metric for binary (i.e., positive or negative) classification problems such as [Logistic Regression](../binaryclass/a9a_lr.html).

Binary classifiers generally predict how likely a sample is to be positive by computing probability. Ultimately, we can evaluate the classifiers by comparing the probabilities with truth positive/negative labels.

Now we assume that there is a table which contains predicted scores (i.e., probabilities) and truth labels as follows:

| probability<br/>(predicted score) | truth label |
|:---:|:---:|
| 0.5 | 0 |
| 0.3 | 1 |
| 0.2 | 0 |
| 0.8 | 1 |
| 0.7 | 1 |

Once the rows are sorted by the probabilities in a descending order, AUC gives a metric based on how many positive (`label=1`) samples are ranked higher than negative (`label=0`) samples. If many positive rows get larger scores than negative rows, AUC would be large, and hence our classifier would perform well.

# Compute AUC on Hivemall

In Hivemall, a function `auc(double score, int label)` provides a way to compute AUC for pairs of probability and truth label.

## Sequential AUC computation on a single node

For instance, the following query computes AUC of the table which was shown above:

```sql
with data as (
  select 0.5 as prob, 0 as label
  union all
  select 0.3 as prob, 1 as label
  union all
  select 0.2 as prob, 0 as label
  union all
  select 0.8 as prob, 1 as label
  union all
  select 0.7 as prob, 1 as label
)
select 
  auc(prob, label) as auc
from (
  select prob, label
  from data
  ORDER BY prob DESC
) t;
```

This query returns `0.83333` as AUC.

Since AUC is a metric based on ranked probability-label pairs as mentioned above, input data (rows) needs to be ordered by scores in a descending order.

## Parallel approximate AUC computation

Meanwhile, Hive's `distribute by` clause allows you to compute AUC in parallel: 

```sql
with data as (
  select 0.5 as prob, 0 as label
  union all
  select 0.3 as prob, 1 as label
  union all
  select 0.2 as prob, 0 as label
  union all
  select 0.8 as prob, 1 as label
  union all
  select 0.7 as prob, 1 as label
)
select 
  auc(prob, label) as auc
from (
  select prob, label
  from data
  DISTRIBUTE BY floor(prob / 0.2)
  SORT BY prob DESC
) t;
```

Note that `floor(prob / 0.2)` means that the rows are distributed to 5 bins for the AUC computation because the column `prob` is in a [0, 1] range.

# Difference between AUC and Logarithmic Loss

Hivemall has another metric called [Logarithmic Loss](stat_eval.html#logarithmic-loss) for binary classification. Both AUC and Logarithmic Loss compute scores for probability-label pairs. 

Score produced by AUC is a relative metric based on sorted pairs. On the other hand, Logarithmic Loss simply gives a metric by comparing probability with its truth label one-by-one.

To give an example, `auc(prob, label)` and `logloss(prob, label)` respectively returns `0.83333` and `0.54001` in the above case. Note that larger AUC and smaller Logarithmic Loss are better.
