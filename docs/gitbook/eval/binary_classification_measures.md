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

# Binary problems

Binary classification problem is predicting the label given two categorized dataset.

Hivemall provides some tutorials to deal with binary classification problems as follows:

- [Online advertisement click prediction](../binaryclass/general.html)
- [News classification](../binaryclass/news20_dataset.html)

This page focuses on evaluation of the results from such binary classification problems.

# Examples

For the metrics explanation, this page introduces toy example data and two metrics.

## Data

The following table shows the sample of binary classification's prediction.
In this case, `1` means positive label and `0` means negative label.
Left column includes supervised label data,
Right column includes are predicted label by a binary classifier.

| truth label| predicted label |
|:---:|:---:|
| 1 | 0 |
| 0 | 1 |
| 0 | 0 |
| 1 | 1 |
| 0 | 1 |

## Preliminary metrics

Some evaluation metrics are calculated based on 4 values:

- True Positive: truth label is positive and predicted label is also positive
- True Negative: truth label is negative and predicted label is also negative
- False Positive: truth label is negative but predicted label is positive
- False Negative: truth label is positive but predicted label is negative

In this example, we can obtain those values:

- True Positive: 1
- True Negative: 1
- False Positive: 2
- False Negative: 1

### Recall

Recall indicates the true positive rate in truth positive labels.
The value is computed by the following equation:

$$
\mathrm{recall} = \frac{\mathrm{\#true\ positive}}{\mathrm{\#true\ positive} + \mathrm{\#false\ negative}}
$$

In the previous example, $$\mathrm{precision} = \frac{1}{2}$$

### Precision

Precision indicates the true positive rate in positive predictive labels.
The value is computed by the following equation:

$$
\mathrm{precision} = \frac{\mathrm{\#true\ positive}}{\mathrm{\#true\ positive} + \mathrm{\#false\ positive}}
$$

In the previous example, $$\mathrm{precision} = \frac{1}{3}$$

# Metrics

## F1-score

F1-score is the harmonic mean of recall and precision.

The value is computed by the following equation:

$$
\mathrm{F}_1 = 2 \frac{\mathrm{precision} * \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$


The Following query shows the example to obtain F1-score.
Each row value has the same type in {int, boolean}.
If row value's type is `int`, `1` is considered as the positive label, and `-1` or `0` is considered as the negative label.

```sql
WITH data as (
  select 1 as truth, 0 as predicted
union all
  select 0 as truth, 1 as predicted
union all
  select 0 as truth, 0 as predicted
union all
  select 1 as truth, 1 as predicted
union all
  select 0 as truth, 1 as predicted
)
select
  f1score(truth, predicted)
from data
;

-- 0.4;
```

## F-measure

F-measure is generalized F1-score and the weighted harmonic mean of recall and precision.

F-measure is computed by the following equation:

$$
\mathrm{F}_{\beta} = (1+\beta^2) \frac{\mathrm{precision} * \mathrm{recall}}{\beta^2 \mathrm{precision} + \mathrm{recall}}
$$

$$\beta$$ is the parameter to determine the weight of precision.
So, F1-score is the special case of F-measure given $$\beta=1$$.

If $$\beta$$ is larger positive value than `1.0`, F-measure reaches to recall.
On the other hand,
if $$\beta$$ is smaller positive value than `1.0`, F-measure reaches to precision.

The following query shows the example to obtain F-measure with $$\beta=2$$.

```sql
WITH data as (
  select 1 as truth, 0 as predicted
union all
  select 0 as truth, 1 as predicted
union all
  select 0 as truth, 0 as predicted
union all
  select 1 as truth, 1 as predicted
union all
  select 0 as truth, 1 as predicted
)
select
  fmeasure(truth, predicted, 2.)
from data
;

-- 0.45454545454545453;
```

When $$\beta$$, the third argument, is omitted, hivemall calculates F-measure with $$\beta=1$$ (: equivalent to F1-score).
