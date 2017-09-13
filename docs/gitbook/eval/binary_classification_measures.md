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

Binary classification is a task to predict a label of each data given two categories.

Hivemall provides several tutorials to deal with binary classification problems as follows:

- [Online advertisement click prediction](../binaryclass/general.html)
- [News classification](../binaryclass/news20_dataset.html)

This page focuses on the evaluation of such binary classification problems.
If your classifier outputs probability rather than 0/1 label, evaluation based on [Area Under the ROC Curve](./auc.md) would be more appropriate.


# Example

This page introduces toy example data and two metrics for explanation.

## Data

The following table shows examples of binary classification's prediction.

| truth label| predicted label | description |
|:---:|:---:|:---:|
| 1 | 0 |False Negative|
| 0 | 1 |False Positive|
| 0 | 0 |True Negative|
| 1 | 1 |True Positive|
| 0 | 1 |False Positive|
| 0 | 0 |True Negative|

In this case, `1` means positive label and `0` means negative label.
The leftmost column shows truth labels, and center column includes predicted labels.

## Preliminary metrics

Some evaluation metrics are calculated based on 4 values:

- True Positive (TP): truth label is positive and predicted label is also positive
- True Negative (TN): truth label is negative and predicted label is also negative
- False Positive (FP): truth label is negative but predicted label is positive
- False Negative (FN): truth label is positive but predicted label is negative

`TR` and `TN` represent correct classification, and `FP` and `FN` illustrate incorrect ones.

In this example, we can obtain those values:

- TP: 1
- TN: 2
- FP: 2
- FN: 1

if you want to know about those metrics, Wikipedia provides [more detail information](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

### Recall

Recall indicates the true positive rate in truth positive labels.
The value is computed by the following equation:

$$
\mathrm{recall} = \frac{\mathrm{\#TP}}{\mathrm{\#TP} + \mathrm{\#FN}}
$$

In the previous example, $$\mathrm{precision} = \frac{1}{2}$$.

### Precision

Precision indicates the true positive rate in positive predictive labels.
The value is computed by the following equation:

$$
\mathrm{precision} = \frac{\mathrm{\#TP}}{\mathrm{\#TP} + \mathrm{\#FP}}
$$

In the previous example, $$\mathrm{precision} = \frac{1}{3}$$.

# Metrics

To use metrics examples, please create the following table.

```sql
create table data as 
  select 1 as truth, 0 as predicted
union all
  select 0 as truth, 1 as predicted
union all
  select 0 as truth, 0 as predicted
union all
  select 1 as truth, 1 as predicted
union all
  select 0 as truth, 1 as predicted
union all
  select 0 as truth, 0 as predicted
;
```

## F1-score

F1-score is the harmonic mean of recall and precision.
F1-score is computed by the following equation:

$$
\mathrm{F}_1 = 2 \frac{\mathrm{precision} * \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}}
$$

Hivemall's `fmeasure` function provides the option which can switch `micro`(default) or `binary` by passing `average` argument.


> #### Caution
> Hivemall also provides `f1score` function, but it is old function to obtain F1-score. The value of `f1score` is based on set operation. So, we recommend to use `fmeasure` function to get F1-score based on this article.

You can learn more about this from the following external resource:

- [scikit-learn's F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)


### Micro average

If `micro` is passed to `average`, 
recall and precision are modified to consider True Negative.
So, micro f1score are calculated by those modified recall and precision.

$$
\mathrm{recall} = \frac{\mathrm{\#TP} + \mathrm{\#TN}}{\mathrm{\#TP} + \mathrm{\#FN} + \mathrm{\#TN}}
$$

$$
\mathrm{precision} = \frac{\mathrm{\#TP} + \mathrm{\#TN}}{\mathrm{\#TP} + \mathrm{\#FP} + \mathrm{\#TN}}
$$

If `average` argument is omitted, `fmeasure` use default value: `'-average micro'`.

The following query shows the example to obtain F1-score.
Each row value has the same type (`int` or `boolean`).
If row value's type is `int`, `1` is considered as the positive label, and `-1` or `0` is considered as the negative label.


```sql
select fmeasure(truth, predicted, '-average micro') from data;
```

> 0.5


It should be noted that, since the old `f1score(truth, predicted)` function simply counts the number of "matched" elements between `truth` and `predicted`, the above query is equivalent to:


```sql
select f1score(array(truth), array(predicted)) from data;
```

### Binary average

If `binary` is passed to `average`, `True Negative` samples are ignored to get F1-score.

The following query shows the example to obtain F1-score with binary average.
```sql
select fmeasure(truth, predicted, '-average binary') from data;
```

> 0.4


## F-measure

F-measure is generalized F1-score and the weighted harmonic mean of recall and precision.
F-measure is computed by the following equation:

$$
\mathrm{F}_{\beta} = (1+\beta^2) \frac{\mathrm{precision} * \mathrm{recall}}{\beta^2 \mathrm{precision} + \mathrm{recall}}
$$

$$\beta$$ is the parameter to determine the weight of precision.
So, F1-score is the special case of F-measure given $$\beta=1$$.

If $$\beta$$ is larger positive value than `1.0`, F-measure reaches recall.
On the other hand,
if $$\beta$$ is smaller positive value than `1.0`, F-measure reaches precision.

If $$\beta$$ is omitted, hivemall calculates F-measure with $$\beta=1$$ (: equivalent to F1-score).

Hivemall's `fmeasure` function also provides the option which can switch `micro`(default) or `binary` by passing `average` argument.


The following query shows the example to obtain F-measure with $$\beta=2$$ and micro average.

```sql
select fmeasure(truth, predicted, '-beta 2. -average micro') from data;
```

> 0.5

The following query shows the example to obtain F-measure with $$\beta=2$$ and binary average.

```sql
select fmeasure(truth, predicted, '-beta 2. -average binary') from data;
```

> 0.45454545454545453

You can learn more about this from the following external resource:

- [scikit-learn's FMeasure](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
