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

[Feature Selection](https://en.wikipedia.org/wiki/Feature_selection) is the process of selecting a subset of relevant features for use in model construction. 

It is a useful technique to 1) improve prediction results by omitting redundant features, 2) to shorten training time, and 3) to know important features for prediction.

*Note: This feature is supported from Hivemall v0.5-rc.1 or later.*

<!-- toc -->

# Supported Feature Selection algorithms

* Chi-square (Chi2)
    * In statistics, the $$\chi^2$$ test is applied to test the independence of two even events. Chi-square statistics between every feature variable and the target variable can be applied to Feature Selection. Refer [this article](https://nlp.stanford.edu/IR-book/html/htmledition/feature-selectionchi2-feature-selection-1.html) for Mathematical details.
* Signal Noise Ratio (SNR)
    * The Signal Noise Ratio (SNR) is a univariate feature ranking metric, which can be used as a feature selection criterion for binary classification problems. SNR is defined as $$|\mu_{1} - \mu_{2}| / (\sigma_{1} + \sigma_{2})$$, where $$\mu_{k}$$ is the mean value of the variable in classes $$k$$, and $$\sigma_{k}$$ is the standard deviations of the variable in classes $$k$$. Clearly, features with larger SNR are useful for classification.

# Usage

##  Feature Selection based on Chi-square test

``` sql
CREATE TABLE input (
  X array<double>, -- features
  Y array<int> -- binarized label
);
 
set hivevar:k=2;

WITH stats AS (
  SELECT
    transpose_and_dot(Y, X) AS observed, -- array<array<double>>, shape = (n_classes, n_features)
    array_sum(X) AS feature_count, -- n_features col vector, shape = (1, array<double>)
    array_avg(Y) AS class_prob -- n_class col vector, shape = (1, array<double>)
  FROM
    input
),
test AS (
  SELECT
    transpose_and_dot(class_prob, feature_count) AS expected -- array<array<double>>, shape = (n_class, n_features)
  FROM
    stats
),
chi2 AS (
  SELECT
    chi2(r.observed, l.expected) AS v -- struct<array<double>, array<double>>, each shape = (1, n_features)
  FROM
    test l
    CROSS JOIN stats r
)
SELECT
  select_k_best(l.X, r.v.chi2, ${k}) as features -- top-k feature selection based on chi2 score
FROM
  input l
  CROSS JOIN chi2 r;
```

## Feature Selection based on Signal Noise Ratio (SNR)

``` sql
CREATE TABLE input (
  X array<double>, -- features
  Y array<int> -- binarized label
);

set hivevar:k=2;

WITH snr AS (
  SELECT snr(X, Y) AS snr -- aggregated SNR as array<double>, shape = (1, #features)
  FROM input
)
SELECT 
  select_k_best(X, snr, ${k}) as features
FROM
  input
  CROSS JOIN snr;
```

# Function signatures

### [UDAF] `transpose_and_dot(X::array<number>, Y::array<number>)::array<array<double>>`

##### Input

| `array<number>` X | `array<number>` Y |
| :-: | :-: |
| a row of matrix | a row of matrix |

##### Output

| `array<array<double>>` dot product |
| :-: |
| `dot(X.T, Y)` of shape = (X.#cols, Y.#cols) |

### [UDF] `select_k_best(X::array<number>, importance_list::array<number>, k::int)::array<double>`

##### Input

| `array<number>` X | `array<number>` importance_list | `int` k |
| :-: | :-: | :-: |
| feature vector | importance of each feature | the number of features to be selected |

##### Output

| `array<array<double>>` k-best features |
| :-: |
| top-k elements from feature vector `X` based on importance list |

### [UDF] `chi2(observed::array<array<number>>, expected::array<array<number>>)::struct<array<double>, array<double>>`

##### Input

| `array<number>` observed | `array<number>` expected |
| :-: | :-: |
| observed features | expected features `dot(class_prob.T, feature_count)` |

Both of `observed` and `expected` have a shape `(#classes, #features)`

##### Output

| `struct<array<double>, array<double>>` importance_list |
| :-: |
| chi2-value and p-value for each feature |

### [UDAF] `snr(X::array<number>, Y::array<int>)::array<double>`

##### Input

| `array<number>` X | `array<int>` Y |
| :-: | :-: |
| feature vector | one hot label |

##### Output

| `array<double>` importance_list |
| :-: |
| Signal Noise Ratio for each feature |

