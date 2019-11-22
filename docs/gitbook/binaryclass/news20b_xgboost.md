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

In this tutorial, we build a binary classification model using XGBoost.

<!-- toc -->

## Feature Vector format for XGBoost

For feature vector, `train_xgboost` takes a sparse vector format (`array<string>`) or a dense vector format (`array<double>`).
In the feature vector, each feature takes a LIBSVM format:

```
feature ::= <index>:<weight>

index ::= <Non-negative INT> (e.g., 0,1,2,...)
weight ::= <DOUBLE>
```

> #### Note
> Unlike the original libsvm format, it's not needed to sort a feature vector by ansceding order of feature index.

Target label format of binary classification follows [this rule](http://hivemall.apache.org/userguide/getting_started/input-format.html#label-format-in-binary-classification). Please refer [xgboost document](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html) as well.

## Label format in Binary Classification

The label must be an INT typed column and the values are positive (+1) or negative (-1) as follows:

```
<label> ::= 1 | -1
```

Alternatively, you can use the following format that represents 1 for a positive example and 0 for a negative example:

```
<label> ::= 0 | 1
```

## Usage and Hyperparameters

You can find hyperparameters and it's default setting by running the following query:

```sql
select train_xgboost();

usage: train_xgboost(array<string|double> features, int|double target [,
       string options]) - Returns a relation consists of <string model_id,
       array<string> pred_model> [-alpha <arg>] [-base_score <arg>]
       [-booster <arg>] [-colsample_bylevel <arg>] [-colsample_bynode
       <arg>] [-colsample_bytree <arg>] [-disable_default_eval_metric
       <arg>] [-eta <arg>] [-eval_metric <arg>] [-feature_selector <arg>]
       [-gamma <arg>] [-grow_policy <arg>] [-lambda <arg>] [-lambda_bias
       <arg>] [-max_bin <arg>] [-max_delta_step <arg>] [-max_depth <arg>]
       [-max_leaves <arg>] [-maximize_evaluation_metrics <arg>]
       [-min_child_weight <arg>] [-normalize_type <arg>] [-num_class
       <arg>] [-num_early_stopping_rounds <arg>] [-num_feature <arg>]
       [-num_parallel_tree <arg>] [-num_pbuffer <arg>] [-num_round <arg>]
       [-objective <arg>] [-one_drop <arg>] [-process_type <arg>]
       [-rate_drop <arg>] [-refresh_leaf <arg>] [-sample_type <arg>]
       [-scale_pos_weight <arg>] [-seed <arg>] [-silent <arg>]
       [-sketch_eps <arg>] [-skip_drop <arg>] [-subsample <arg>] [-top_k
       <arg>] [-tree_method <arg>] [-tweedie_variance_power <arg>]
       [-updater <arg>] [-validation_ratio <arg>] [-verbosity <arg>]
 -alpha,--reg_alpha <arg>             L1 regularization term on weights.
                                      Increasing this value will make
                                      model more conservative. [default:
                                      0.0]
 -base_score <arg>                    Initial prediction score of all
                                      instances, global bias [default:
                                      0.5]
 -booster <arg>                       Set a booster to use, gbtree or
                                      gblinear or dart. [default: gbree]
 -colsample_bylevel <arg>             Subsample ratio of columns for each
                                      level [default: 1.0]
 -colsample_bynode <arg>              Subsample ratio of columns for each
                                      node [default: 1.0]
 -colsample_bytree <arg>              Subsample ratio of columns when
                                      constructing each tree [default:
                                      1.0]
 -disable_default_eval_metric <arg>   NFlag to disable default metric. Set
                                      to >0 to disable. [default: 0]
 -eta,--learning_rate <arg>           Step size shrinkage used in update
                                      to prevents overfitting [default:
                                      0.3]
 -eval_metric <arg>                   Evaluation metrics for validation
                                      data. A default metric is assigned
                                      according to the objective:
                                      - rmse: for regression
                                      - error: for classification
                                      - map: for ranking
                                      For a list of valid inputs, see
                                      XGBoost Parameters.
 -feature_selector <arg>              Feature selection and ordering
                                      method. [Choices: cyclic (default),
                                      shuffle, random, greedy, thrifty]
 -gamma,--min_split_loss <arg>        Minimum loss reduction required to
                                      make a further partition on a leaf
                                      node of the tree. [default: 0.0]
 -grow_policy <arg>                   Controls a way new nodes are added
                                      to the tree. Currently supported
                                      only if tree_method is set to hist.
                                      [default: depthwise, Choices:
                                      depthwise, lossguide]
 -lambda,--reg_lambda <arg>           L2 regularization term on weights.
                                      Increasing this value will make
                                      model more conservative. [default:
                                      1.0 for gbtree, 0.0 for gblinear]
 -lambda_bias <arg>                   L2 regularization term on bias
                                      [default: 0.0]
 -max_bin <arg>                       Maximum number of discrete bins to
                                      bucket continuous features. Only
                                      used if tree_method is set to hist.
                                      [default: 256]
 -max_delta_step <arg>                Maximum delta step we allow each
                                      tree's weight estimation to be
                                      [default: 0]
 -max_depth <arg>                     Max depth of decision tree [default:
                                      6]
 -max_leaves <arg>                    Maximum number of nodes to be added.
                                      Only relevant when
                                      grow_policy=lossguide is set.
                                      [default: 0]
 -maximize_evaluation_metrics <arg>   Maximize evaluation metrics
                                      [default: false]
 -min_child_weight <arg>              Minimum sum of instance weight
                                      (hessian) needed in a child
                                      [default: 1.0]
 -normalize_type <arg>                Type of normalization algorithm.
                                      [Choices: tree (default), forest]
 -num_class <arg>                     Number of classes to classify
 -num_early_stopping_rounds <arg>     Minimum rounds required for early
                                      stopping [default: 0]
 -num_feature <arg>                   Feature dimension used in boosting
                                      [default: set automatically by
                                      xgboost]
 -num_parallel_tree <arg>             Number of parallel trees constructed
                                      during each iteration. This option
                                      is used to support boosted random
                                      forest. [default: 1]
 -num_pbuffer <arg>                   Size of prediction buffer [default:
                                      set automatically by xgboost]
 -num_round,--iters <arg>             Number of boosting iterations
                                      [default: 10]
 -objective <arg>                     Specifies the learning task and the
                                      corresponding learning objective.
                                      Examples: reg:linear, reg:logistic,
                                      multi:softmax. For a full list of
                                      valid inputs, refer to XGBoost
                                      Parameters. [default: reg:linear]
 -one_drop <arg>                      When this flag is enabled, at least
                                      one tree is always dropped during
                                      the dropout. 0 or 1. [default: 0]
 -process_type <arg>                  A type of boosting process to run.
                                      [Choices: default, update]
 -rate_drop <arg>                     Dropout rate in range [0.0, 1.0].
                                      [default: 0.0]
 -refresh_leaf <arg>                  This is a parameter of the refresh
                                      updater plugin. When this flag is 1,
                                      tree leafs as well as tree nodes’
                                      stats are updated. When it is 0,
                                      only node stats are updated.
                                      [default: 1]
 -sample_type <arg>                   Type of sampling algorithm.
                                      [Choices: uniform (default),
                                      weighted]
 -scale_pos_weight <arg>              ontrol the balance of positive and
                                      negative weights, useful for
                                      unbalanced classes. A typical value
                                      to consider: sum(negative instances)
                                      / sum(positive instances) [default:
                                      1.0]
 -seed <arg>                          Random number seed. [default: 43]
 -silent <arg>                        Deprecated. Please use verbosity
                                      instead. 0 means printing running
                                      messages, 1 means silent mode
                                      [default: 1]
 -sketch_eps <arg>                    This roughly translates into O(1 /
                                      sketch_eps) number of bins.
                                      Compared to directly select number
                                      of bins, this comes with theoretical
                                      guarantee with sketch accuracy.
                                      Only used for tree_method=approx.
                                      Usually user does not have to tune
                                      this.  [default: 0.03]
 -skip_drop <arg>                     Probability of skipping the dropout
                                      procedure during a boosting
                                      iteration in range [0.0, 1.0].
                                      [default: 0.0]
 -subsample <arg>                     Subsample ratio of the training
                                      instance in range (0.0,1.0]
                                      [default: 1.0]
 -top_k <arg>                         The number of top features to select
                                      in greedy and thrifty feature
                                      selector. The value of 0 means using
                                      all the features. [default: 0]
 -tree_method <arg>                   The tree construction algorithm used
                                      in XGBoost. [default: auto, Choices:
                                      auto, exact, approx, hist]
 -tweedie_variance_power <arg>        Parameter that controls the variance
                                      of the Tweedie distribution in range
                                      [1.0, 2.0]. [default: 1.5]
 -updater <arg>                       A comma-separated string that
                                      defines the sequence of tree
                                      updaters to run. For a full list of
                                      valid inputs, please refer to
                                      XGBoost Parameters. [default:
                                      'grow_colmaker,prune' for gbtree,
                                      'shotgun' for gblinear]
 -validation_ratio <arg>              Validation ratio in range [0.0,1.0]
                                      [default: 0.2]
 -verbosity <arg>                     Verbosity of printing messages.
                                      Choices: 0 (silent), 1 (warning), 2
                                      (info), 3 (debug). [default: 0]
```

Objective function `-objective` SHOULD be specified though `-objective reg:linear` is used for Objective function by the default.
For the full list of objective functions, please refer [this xgboost v0.90 documentation](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters).

The following objectives would widely be used for regression, binary classication, and multiclass classication, respectively.

- `reg:squarederror` regression with squared loss.
- `binary:logistic` logistic regression for binary classification, output probability.
- `binary:hinge` hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
- `multi:softmax` set XGBoost to do multiclass classification using the softmax objective, you also need to set `num_class` (number of classes).
- `multi:softprob` same as softmax, but output a vector of `ndata * nclass`, which can be further reshaped to `ndata * nclass` matrix. The result contains predicted probability of each data point belonging to each class.

Other hyperparameters better to be tuned are:

- `-booster gbree` Which booster to use. The default gbtree (Gradient Boosting Trees) would be fine for most cases. Can be `gbtree`, `gblinear` or `dart`; gbtree and dart use tree based models while gblinear uses linear functions.
- `-eta 0.1` The learning rate, 0.3 by the default. 0.05, 0.1, 0.3 are worth trying.
- `-max_depth 6` The maximum depth of the tree. The default value 6 would be fine for most case. Recommended value range is 5-10.
- `-num_class 3` The number of classes MUST be specified for multiclass classification (i.e., `-objective multi:softmax` or `-objective multi:softprob`)
- `-num_round 10` The number of rounds for boosting. 10 or more would be preferred.
- `-num_early_stopping_rounds 3` The number of rounds required for early stopping. Without specifying `-num_early_stopping_rounds`, no early stopping is NOT carried. When `-num_round=100` and `-num_early_stopping_rounds=5`, traning could be early stopped at 15th iteration if there is no evaluation result greater than the 10th iteration's (best one). Early stopping 3 or so would be preferred. 
- `-validation_ratio 0.2` The ratio data used for validation (early stopping). 0.2 would be enough for most cases. Note that 80% data is used for training when `validation_ratio 0.2` is set.

You can find the underlying XGBoost version by:

```sql
select xgboost_version();
> 0.90
```

## Training

`train_xgboost` UDTF is used for training. 

The function signature is `train_xgboost(array<string|double> features, double target [,string options])` and it returns a prediction model as a relation consist of `<string model_id, array<string> pred_model>`.

```sql
-- explicitly use 3 reducers
-- set mapred.reduce.tasks=3;

drop table xgb_lr_model;
create table xgb_lr_model as
select 
  train_xgboost(features, label, '-objective binary:logistic -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select features, label
  from news20b_train
  cluster by rand(43) -- shuffle data to reducers
) shuffled;

drop table xgb_hinge_model;
create table xgb_hinge_model as
select 
  train_xgboost(features, label, '-objective binary:hinge -num_round 10 -num_early_stopping_rounds 3') 
    as (model_id, model)
from (
  select features, label
  from news20b_train
  cluster by rand(43) -- shuffle data to reducers
) shuffled;
```

> #### Caution
> `cluster by rand()` is NOT required when training data is small and a single task is launched for XGBoost training.
> `cluster by rand()` shuffles data at random and divided it for multiple XGBoost instances.

## prediction

```sql
drop table xgb_lr_predicted;
create table xgb_lr_predicted 
as
select
  rowid, 
  array_avg(predicted) as predicted,
  avg(predicted[0]) as prob
from (
  select
    -- fast predictition by xgboost-predictor-java (https://github.com/komiya-atsushi/xgboost-predictor-java/)
    xgboost_predict(rowid, features, model_id, model) as (rowid, predicted)
    -- predict by  xgboost4j (https://xgboost.readthedocs.io/en/stable/jvm/)
    -- xgboost_batch_predict(rowid, features, model_id, model) as (rowid, predicted)
  from
    -- for each model l 
    --   for each test r
    --     predict
    xgb_lr_model l
    LEFT OUTER JOIN news20b_test r 
) t
group by rowid;

drop table xgb_hinge_predicted;
create table xgb_hinge_predicted 
as
select
  rowid,
  -- voting
  -- if(sum(if(predicted[0]=1,1,0)) > sum(if(predicted[0]=0,1,0)),1,-1) as predicted
  majority_vote(if(predicted[0]=1, 1, -1)) as predicted
from (
  select
    -- binary:hinge is not supported in xgboost_predict
    -- binary:hinge returns [1.0] or [0.0] for predicted
    xgboost_batch_predict(rowid, features, model_id, model) 
      as (rowid, predicted)
  from
    -- for each model l 
    --   for each test r
    --     predict
    xgb_hinge_model l
    LEFT OUTER JOIN news20b_test r 
) t
group by
  rowid
```

You can find the function signature of `xgboost_predict` by

```sql
select xgboost_predict();

usage: xgboost_predict(PRIMITIVE rowid, array<string|double> features,
       string model_id, array<string> pred_model [, string options]) -
       Returns a prediction result as (string rowid, array<double>
       predicted)

select xgboost_batch_predict();

usage: xgboost_batch_predict(PRIMITIVE rowid, array<string|double>
       features, string model_id, array<string> pred_model [, string
       options]) - Returns a prediction result as (string rowid,
       array<double> predicted) [-batch_size <arg>]
 -batch_size <arg>   Number of rows to predict together [default: 128]
```

> #### Caution
> `xgboost_predict` outputs probability for `-objective binary:logistic` while 0/1 is resulted for `-objective binary:hinge`.
> 
> `xgboost_predict` only support the following models and objectives because it uses [xgboost-predictor-java](https://github.com/komiya-atsushi/xgboost-predictor-java):
> Models: {gblinear, gbtree, dart}
> Objective functions: {binary:logistic, binary:logitraw, multi:softmax, multi:softprob, reg:linear, reg:squarederror, rank:pairwise}
> 
> For other models and objectives, please use `xgboost_batch_predict` that uses [xgboost4j](https://xgboost.readthedocs.io/en/stable/jvm/) insead.

## evaluation

```sql
WITH submit as (
  select 
    t.label as actual, 
    -- probability thresholding by 0.5
    if(p.prob > 0.5,1,-1)  as predicted
  from 
    news20b_test t 
    JOIN xgb_lr_predicted p
      on (t.rowid = p.rowid)
)
select 
  sum(if(actual = predicted, 1, 0)) / count(1) as accuracy
from
  submit;
```

> 0.8372698158526821 (logistic loss)

```sql
WITH submit as (
  select 
   t.label as actual, 
   p.predicted
  from 
    news20b_test t 
    JOIN xgb_hinge_predicted p
      on (t.rowid = p.rowid)
)
select 
  sum(if(actual=predicted,1,0)) / count(1) as accuracy
from
  submit;
```

> 0.7752201761409128 (hinge loss)

