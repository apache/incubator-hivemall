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

Hivemall has some metrics to evaluate binary classification model for some specific tasks, 
e.g. classify spam mail or not, [predict whether a user will click online advertisement or not](../binaryclass/general.html).

The table below shows the sample of binary classification's prediction.
In this table, `1` means positive label and `0` means negative label.
Left column includes supervised label data, 
Right column includes are predicted labey by a binary classifier.

| truth label| predicted label |
|:---:|:---:|
| 1 | 0 |
| 0 | 1 |
| 0 | 0 |
| 1 | 1 |
| 0 | 1 |

Some evaluation metrics are calculated based on four values:

- True Positive (TP): trurh label is positive and predicted label is also positive
- True Negative (TN): trurh label is negative and predicted label is also negative
- False Positive (FP): trurh label is negative but predicted label is positive
- False Negative (FN): trurh label is positive but predicted label is negative

So, we can get those values from the table:

- TP: 1
- TN: 2
- FP: 1
- FN: 1

## Recall

Recall indicates the true positive rate in truth positive labels.
The value is computed by the following equation:

$$
\mathrm{precision} = \frac{\mathrm{\#true\ positive}}{\mathrm{\#true\ positive} + \mathrm{\#false\ negative}}
$$


## Precision

$$
\mathrm{precision} = \frac{\mathrm{\#true\ positive}}{\mathrm{\#true\ positive} + \mathrm{\#false\ positive}}
$$

## F1-score

$$
\mathrm{f}_1 = 2 \frac{\mathrm{recision} * \mathrm{recall}}{\mathrm{recision} + \mathrm{recall}}
$$

## F-measure

F-measure is feneralize F1score.
F1-score is special case of F-measure when $$\beta=1$$.

$$
\mathrm{f}_{\beta} = (1+\beta^2) \frac{\mathrm{recision} * \mathrm{recall}}{\beta^2 \mathrm{recision} + \mathrm{recall}}
$$
