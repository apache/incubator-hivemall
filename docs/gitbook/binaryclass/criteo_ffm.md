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

[Field-aware factorization machines](https://dl.acm.org/citation.cfm?id=2959134) (FFM) is a factorization model which has been used by the [#1 solution](https://www.kaggle.com/c/criteo-display-ad-challenge/discussion/10555) of the Criteo competition.

This page guides you to try the factorization technique with Hivemall's `train_ffm` and `ffm_predict` UDFs.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5.1 or later.

# Preprocess data and convert into LIBFFM format

Since FFM is a relatively complex factor-based model which requires us to spend a significant amount of time for feature engineering, preprocessing data outside of Hive can be a reasonable option.

You can again use the repository **[takuti/criteo-ffm](https://github.com/takuti/criteo-ffm)** cloned in the [data preparation guide](criteo_dataset.md) to preprocess the data as the winning solution did:

```sh
cd criteo-ffm
# create the CSV files `tr.csv` and `te.csv`
make preprocess
```

Task `make preprocess` executes some Python scripts which are originally taken from [guestwalk/kaggle-2014-criteo](https://github.com/guestwalk/kaggle-2014-criteo) and [chenhuang-learn/ffm](https://github.com/chenhuang-learn/ffm).

Eventually, you will obtain the following files in so-called LIBFFM format:

- `tr.ffm` - Labeled training samples
  - `tr.sp` - 80% of the labeled training samples randomly picked from `tr.ffm`
  - `va.sp` - Remaining 20% of samples for evaluation
- `te.ffm` - Unlabeled test samples

```
<label> <field1>:<feature1>:<value1> <field2>:<feature2>:<value2> ...
.
.
.
```

See [LIBFFM official README](https://github.com/guestwalk/libffm) for detail.

In order to evaluate the accuracy of prediction at the end of this tutorial, later sections use `tr.sp` and `va.sp`.

# Insert preprocessed data into tables

Create new tables used by the FFM UDFs:

```sh
hadoop fs -put tr.sp /criteo/ffm/train
hadoop fs -put va.sp /criteo/ffm/test
```

```sql
use criteo;
```

```sql
DROP TABLE IF EXISTS train_ffm;
CREATE EXTERNAL TABLE train_ffm (
  label int,
  -- quantitative features
  i1 string,i2 string,i3 string,i4 string,i5 string,i6 string,i7 string,i8 string,i9 string,i10 string,i11 string,i12 string,i13 string,
  -- categorical features
  c1 string,c2 string,c3 string,c4 string,c5 string,c6 string,c7 string,c8 string,c9 string,c10 string,c11 string,c12 string,c13 string,c14 string,c15 string,c16 string,c17 string,c18 string,c19 string,c20 string,c21 string,c22 string,c23 string,c24 string,c25 string,c26 string
) ROW FORMAT
DELIMITED FIELDS TERMINATED BY ' '
STORED AS TEXTFILE LOCATION '/criteo/ffm/train';
```

```sql
DROP TABLE IF EXISTS test_ffm;
CREATE EXTERNAL TABLE test_ffm (
  label int,
  -- quantitative features
  i1 string,i2 string,i3 string,i4 string,i5 string,i6 string,i7 string,i8 string,i9 string,i10 string,i11 string,i12 string,i13 string,
  -- categorical features
  c1 string,c2 string,c3 string,c4 string,c5 string,c6 string,c7 string,c8 string,c9 string,c10 string,c11 string,c12 string,c13 string,c14 string,c15 string,c16 string,c17 string,c18 string,c19 string,c20 string,c21 string,c22 string,c23 string,c24 string,c25 string,c26 string
) ROW FORMAT
DELIMITED FIELDS TERMINATED BY ' '
STORED AS TEXTFILE LOCATION '/criteo/ffm/test';
```

Vectorize the LIBFFM-formatted features with `rowid`:

```sql
DROP TABLE IF EXISTS train_vectorized;
CREATE TABLE train_vectorized AS
SELECT
  row_number() OVER () AS rowid,
  array(
    i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26
  ) AS features,
  label
FROM
  train_ffm
;
```

```sql
DROP TABLE IF EXISTS test_vectorized;
CREATE TABLE test_vectorized AS
SELECT
  row_number() OVER () AS rowid,
  array(
    i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26
  ) AS features,
  label
FROM
  test_ffm
;
```

# Training

```sql
DROP TABLE IF EXISTS criteo.ffm_model;
CREATE TABLE  criteo.ffm_model (
  model_id int,
  i int,
  Wi float,
  Vi array<float>
);
```

```sql
INSERT OVERWRITE TABLE criteo.ffm_model
SELECT
  train_ffm(
    features,
    label,
    '-init_v random -max_init_value 0.5 -classification -iterations 15 -factors 4 -eta 0.2 -optimizer adagrad -lambda 0.00002'
  )
FROM (
  SELECT
    features, label
  FROM
    criteo.train_vectorized
  CLUSTER BY rand(1)
) t
;
```

The third argument of `train_ffm` accepts a variety of options:

```
hive> SELECT train_ffm(array(), 0, '-help');
usage: train_ffm(array<string> x, double y [, const string options]) -
       Returns a prediction model [-alpha <arg>] [-auto_stop] [-beta
       <arg>] [-c] [-cv_rate <arg>] [-disable_cv] [-enable_norm]
       [-enable_wi] [-eps <arg>] [-eta <arg>] [-eta0 <arg>] [-f <arg>]
       [-feature_hashing <arg>] [-help] [-init_v <arg>] [-int_feature]
       [-iters <arg>] [-l1 <arg>] [-l2 <arg>] [-lambda0 <arg>] [-lambdaV
       <arg>] [-lambdaW0 <arg>] [-lambdaWi <arg>] [-max <arg>] [-maxval
       <arg>] [-min <arg>] [-min_init_stddev <arg>] [-no_norm]
       [-num_fields <arg>] [-opt <arg>] [-p <arg>] [-power_t <arg>] [-seed
       <arg>] [-sigma <arg>] [-t <arg>] [-va_ratio <arg>] [-va_threshold
       <arg>] [-w0]
 -alpha,--alphaFTRL <arg>                     Alpha value (learning rate)
                                              of
                                              Follow-The-Regularized-Reade
                                              r [default: 0.2]
 -auto_stop,--early_stopping                  Stop at the iteration that
                                              achieves the best validation
                                              on partial samples [default:
                                              OFF]
 -beta,--betaFTRL <arg>                       Beta value (a learning
                                              smoothing parameter) of
                                              Follow-The-Regularized-Reade
                                              r [default: 1.0]
 -c,--classification                          Act as classification
 -cv_rate,--convergence_rate <arg>            Threshold to determine
                                              convergence [default: 0.005]
 -disable_cv,--disable_cvtest                 Whether to disable
                                              convergence check [default:
                                              OFF]
 -enable_norm,--l2norm                        Enable instance-wise L2
                                              normalization
 -enable_wi,--linear_term                     Include linear term
                                              [default: OFF]
 -eps <arg>                                   A constant used in the
                                              denominator of AdaGrad
                                              [default: 1.0]
 -eta <arg>                                   The initial learning rate
 -eta0 <arg>                                  The initial learning rate
                                              [default 0.1]
 -f,--factors <arg>                           The number of the latent
                                              variables [default: 5]
 -feature_hashing <arg>                       The number of bits for
                                              feature hashing in range
                                              [18,31] [default: -1]. No
                                              feature hashing for -1.
 -help                                        Show function help
 -init_v <arg>                                Initialization strategy of
                                              matrix V [random,
                                              gaussian](default: 'random'
                                              for regression / 'gaussian'
                                              for classification)
 -int_feature,--feature_as_integer            Parse a feature as integer
                                              [default: OFF]
 -iters,--iterations <arg>                    The number of iterations
                                              [default: 10]
 -l1,--lambda1 <arg>                          L1 regularization value of
                                              Follow-The-Regularized-Reade
                                              r that controls model
                                              Sparseness [default: 0.001]
 -l2,--lambda2 <arg>                          L2 regularization value of
                                              Follow-The-Regularized-Reade
                                              r [default: 0.0001]
 -lambda0,--lambda <arg>                      The initial lambda value for
                                              regularization [default:
                                              0.0001]
 -lambdaV,--lambda_v <arg>                    The initial lambda value for
                                              V regularization [default:
                                              0.0001]
 -lambdaW0,--lambda_w0 <arg>                  The initial lambda value for
                                              W0 regularization [default:
                                              0.0001]
 -lambdaWi,--lambda_wi <arg>                  The initial lambda value for
                                              Wi regularization [default:
                                              0.0001]
 -max,--max_target <arg>                      The maximum value of target
                                              variable
 -maxval,--max_init_value <arg>               The maximum initial value in
                                              the matrix V [default: 0.5]
 -min,--min_target <arg>                      The minimum value of target
                                              variable
 -min_init_stddev <arg>                       The minimum standard
                                              deviation of initial matrix
                                              V [default: 0.1]
 -no_norm,--disable_norm                      Disable instance-wise L2
                                              normalization
 -num_fields <arg>                            The number of fields
                                              [default: 256]
 -opt,--optimizer <arg>                       Gradient Descent optimizer
                                              [default: ftrl, adagrad,
                                              sgd]
 -p,--num_features <arg>                      The size of feature
                                              dimensions [default: -1]
 -power_t <arg>                               The exponent for inverse
                                              scaling learning rate
                                              [default 0.1]
 -seed <arg>                                  Seed value [default: -1
                                              (random)]
 -sigma <arg>                                 The standard deviation for
                                              initializing V [default:
                                              0.1]
 -t,--total_steps <arg>                       The total number of training
                                              examples
 -va_ratio,--validation_ratio <arg>           Ratio of training data used
                                              for validation [default:
                                              0.05f]
 -va_threshold,--validation_threshold <arg>   Threshold to start
                                              validation. At least N
                                              training examples are used
                                              before validation [default:
                                              1000]
 -w0,--global_bias                            Whether to include global
                                              bias term w0 [default: OFF]
```

Note that debug log describes the change of cumulative loss over iterations as follows:

```
Iteration #2 | average loss=0.5407147187026483, current cumulative loss=858.114258581103, previous cumulative loss=1682.1101438997914, change rate=0.48985846040280256, #trainingExamples=1587
Iteration #3 | average loss=0.5105058761578417, current cumulative loss=810.1728254624949, previous cumulative loss=858.114258581103, change rate=0.05586835626980435, #trainingExamples=1587
Iteration #4 | average loss=0.49045915570992393, current cumulative loss=778.3586801116493, previous cumulative loss=810.1728254624949, change rate=0.039268344174200345, #trainingExamples=1587
Iteration #5 | average loss=0.4752751205770395, current cumulative loss=754.2616163557617, previous cumulative loss=778.3586801116493, change rate=0.030958816766109738, #trainingExamples=1587
Iteration #6 | average loss=0.46308523885164105, current cumulative loss=734.9162740575543, previous cumulative loss=754.2616163557617, change rate=0.02564805351182389, #trainingExamples=1587
Iteration #7 | average loss=0.4529012395753083, current cumulative loss=718.7542672060143, previous cumulative loss=734.9162740575543, change rate=0.02199163009727323, #trainingExamples=1587
Iteration #8 | average loss=0.44411358945347845, current cumulative loss=704.8082664626703, previous cumulative loss=718.7542672060143, change rate=0.019403016273636577, #trainingExamples=1587
Iteration #9 | average loss=0.4363264696377158, current cumulative loss=692.450107315055, previous cumulative loss=704.8082664626703, change rate=0.017534072365012268, #trainingExamples=1587
Iteration #10 | average loss=0.4292753045556725, current cumulative loss=681.2599083298522, previous cumulative loss=692.450107315055, change rate=0.01616029641267912, #trainingExamples=1587
Iteration #11 | average loss=0.42277515600757143, current cumulative loss=670.9441725840159, previous cumulative loss=681.2599083298522, change rate=0.015142144165104322, #trainingExamples=1587
Iteration #12 | average loss=0.416689617663307, current cumulative loss=661.2864232316682, previous cumulative loss=670.9441725840159, change rate=0.014394266687126348, #trainingExamples=1587
Iteration #13 | average loss=0.4109140194740033, current cumulative loss=652.1205489052433, previous cumulative loss=661.2864232316682, change rate=0.013860672175351585, #trainingExamples=1587
Iteration #14 | average loss=0.4053667348634373, current cumulative loss=643.317008228275, previous cumulative loss=652.1205489052433, change rate=0.013499866998129951, #trainingExamples=1587
Iteration #15 | average loss=0.3999840450561501, current cumulative loss=634.7746795041102, previous cumulative loss=643.317008228275, change rate=0.013278568131893133, #trainingExamples=1587
Performed 15 iterations of 1,587 training examples on memory (thus 23,805 training updates in total)
```

# Prediction and evaluation

```sql
DROP TABLE IF EXISTS criteo.test_exploded;
CREATE TABLE criteo.test_exploded AS
SELECT
  t1.rowid,
  t2.i,
  t2.j,
  t2.Xi,
  t2.Xj
from
  criteo.test_vectorized t1
  LATERAL VIEW feature_pairs(t1.features, '-ffm') t2 AS i, j, Xi, Xj
;
```

```sql
WITH predicted AS (
  SELECT
    rowid,
    avg(score) AS predicted
  FROM (
    SELECT
      t1.rowid,
      p1.model_id,
      sigmoid(ffm_predict(p1.Wi, p1.Vi, p2.Vi, t1.Xi, t1.Xj)) AS score
    FROM
      criteo.test_exploded t1
      JOIN criteo.ffm_model p1 ON (p1.i = t1.i) -- at least p1.i = 0 and t1.i = 0 exists
      LEFT OUTER JOIN criteo.ffm_model p2 ON (p2.model_id = p1.model_id and p2.i = t1.j)
    WHERE
      p1.Wi is not null OR p2.Vi is not null
    GROUP BY
      t1.rowid, p1.model_id
  ) t
  GROUP BY
    rowid
)
SELECT
  logloss(t1.predicted, t2.label)
FROM
  predicted t1
JOIN
  criteo.test_vectorized t2
  ON t1.rowid = t2.rowid
;
```

> 0.47276208106423234

<br />

> #### Note
> The accuracy varies depending on the random separation of `tr.sp` and `va.sp`.

Notice that LogLoss around 0.45 is reasonable accuracy compared to the [competition leaderboard](https://github.com/guestwalk/libffm) and output from [LIBFFM](https://github.com/guestwalk/libffm).