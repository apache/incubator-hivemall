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

# Summary

## TABLE OF CONTENTS

* [Getting Started](getting_started/README.md)
    * [Installation](getting_started/installation.md)
    * [Install as permanent functions](getting_started/permanent-functions.md)
    * [Input Format](getting_started/input-format.md)
    * [Step-by-Step Tutorial on Supervised Learning](getting_started/tutorial.md)

* [List of Functions](misc/funcs.md)

* [Tips for Effective Hivemall](tips/README.md)
    * [Explicit add_bias() for better prediction](tips/addbias.md)
    * [Use rand_amplify() to better prediction results](tips/rand_amplify.md)
    * [Real-time prediction on RDBMS](tips/rt_prediction.md)
    * [Ensemble learning for stable prediction](tips/ensemble_learning.md)
    * [Mixing models for a better prediction convergence (MIX server)](tips/mixserver.md)
    * [Run Hivemall on Amazon Elastic MapReduce](tips/emr.md)

* [General Hive/Hadoop Tips](tips/general_tips.md)
    * [Adding rowid for each row](tips/rowid.md)
    * [Hadoop tuning for Hivemall](tips/hadoop_tuning.md)

* [Troubleshooting](troubleshooting/README.md)
    * [OutOfMemoryError in training](troubleshooting/oom.md)
    * [SemanticException generate map join task error: Cannot serialize object](troubleshooting/mapjoin_task_error.md)
    * [Asterisk argument for UDTF does not work](troubleshooting/asterisk.md)
    * [The number of mappers is less than input splits in Hadoop 2.x](troubleshooting/num_mappers.md)
    * [Map-side join causes ClassCastException on Tez](troubleshooting/mapjoin_classcastex.md)

## Part II - Generic Features

* [List of Generic Hivemall Functions](misc/generic_funcs.md)
* [Efficient Top-K Query Processing](misc/topk.md)
* [Text Tokenizer](misc/tokenizer.md)
* [Approximate Aggregate Functions](misc/approx.md)

## Part III - Feature Engineering

* [Feature Scaling](ft_engineering/scaling.md)
* [Feature Hashing](ft_engineering/hashing.md)
* [Feature Selection](ft_engineering/selection.md)
* [Feature Binning](ft_engineering/binning.md)
* [Feature Paring](ft_engineering/pairing.md)
    * [Polynomial features](ft_engineering/polynomial.md)
* [Feature Transformation](ft_engineering/ft_trans.md)
    * [Feature vectorization](ft_engineering/vectorization.md)
    * [Quantify non-number features](ft_engineering/quantify.md)
* [TF-IDF Calculation](ft_engineering/tfidf.md)

## Part IV - Evaluation

* [Binary Classification Metrics](eval/binary_classification_measures.md)
    * [Area under the ROC curve](eval/auc.md)
* [Multi-label Classification Metrics](eval/multilabel_classification_measures.md)
* [Regression Metrics](eval/regression.md)
* [Ranking Measures](eval/rank.md)
* [Data Generation](eval/datagen.md)
    * [Logistic Regression data generation](eval/lr_datagen.md)

## Part V - Supervised Learning

* [How Prediction Works](misc/prediction.md)

## Part VI - Binary Classification

* [Binary Classification](binaryclass/general.md)

* [a9a Tutorial](binaryclass/a9a.md)
    * [Data preparation](binaryclass/a9a_dataset.md)
    * [Logistic Regression](binaryclass/a9a_lr.md)
    * [Mini-batch gradient descent](binaryclass/a9a_minibatch.md)

* [News20 Tutorial](binaryclass/news20.md)
    * [Data preparation](binaryclass/news20_dataset.md)
    * [Perceptron, Passive Aggressive](binaryclass/news20_pa.md)
    * [CW, AROW, SCW](binaryclass/news20_scw.md)
    * [AdaGradRDA, AdaGrad, AdaDelta](binaryclass/news20_adagrad.md)
    * [Random Forest](binaryclass/news20_rf.md)

* [KDD2010a Tutorial](binaryclass/kdd2010a.md)
    * [Data preparation](binaryclass/kdd2010a_dataset.md)
    * [PA, CW, AROW, SCW](binaryclass/kdd2010a_scw.md)

* [KDD2010b Tutorial](binaryclass/kdd2010b.md)
    * [Data preparation](binaryclass/kdd2010b_dataset.md)
    * [AROW](binaryclass/kdd2010b_arow.md)

* [Webspam Tutorial](binaryclass/webspam.md)
    * [Data pareparation](binaryclass/webspam_dataset.md)
    * [PA1, AROW, SCW](binaryclass/webspam_scw.md)

* [Kaggle Titanic Tutorial](binaryclass/titanic_rf.md)


## Part VII - Multiclass Classification

* [News20 Multiclass Tutorial](multiclass/news20.md)
    * [Data preparation](multiclass/news20_dataset.md)
    * [Data preparation for one-vs-the-rest classifiers](multiclass/news20_one-vs-the-rest_dataset.md)
    * [PA](multiclass/news20_pa.md)
    * [CW, AROW, SCW](multiclass/news20_scw.md)
    * [Ensemble learning](multiclass/news20_ensemble.md)
    * [one-vs-the-rest classifier](multiclass/news20_one-vs-the-rest.md)

* [Iris Tutorial](multiclass/iris.md)
    * [Data preparation](multiclass/iris_dataset.md)
    * [SCW](multiclass/iris_scw.md)
    * [Random Forest](multiclass/iris_randomforest.md)

## Part VIII - Regression

* [Regression](regression/general.md)

* [E2006-tfidf Regression Tutorial](regression/e2006.md)
    * [Data preparation](regression/e2006_dataset.md)
    * [Passive Aggressive, AROW](regression/e2006_arow.md)

* [KDDCup 2012 Track 2 CTR Prediction Tutorial](regression/kddcup12tr2.md)
    * [Data preparation](regression/kddcup12tr2_dataset.md)
    * [Logistic Regression, Passive Aggressive](regression/kddcup12tr2_lr.md)
    * [Logistic Regression with amplifier](regression/kddcup12tr2_lr_amplify.md)
    * [AdaGrad, AdaDelta](regression/kddcup12tr2_adagrad.md)

## Part IX - Recommendation

* [Collaborative Filtering](recommend/cf.md)
    * [Item-based collaborative filtering](recommend/item_based_cf.md)

* [News20 Related Article Recommendation Tutorial](recommend/news20.md)
    * [Data preparation](multiclass/news20_dataset.md)
    * [LSH/MinHash and Jaccard similarity](recommend/news20_jaccard.md)
    * [LSH/MinHash and brute-force search](recommend/news20_knn.md)
    * [kNN search using b-Bits MinHash](recommend/news20_bbit_minhash.md)

* [MovieLens Movie Recommendation Tutorial](recommend/movielens.md)
    * [Data preparation](recommend/movielens_dataset.md)
    * [Item-based collaborative filtering](recommend/movielens_cf.md)
    * [Matrix Factorization](recommend/movielens_mf.md)
    * [Factorization Machine](recommend/movielens_fm.md)
    * [SLIM for fast top-k recommendation](recommend/movielens_slim.md)
    * [10-fold cross validation (Matrix Factorization)](recommend/movielens_cv.md)

## Part X - Anomaly Detection

* [Outlier Detection using Local Outlier Factor (LOF)](anomaly/lof.md)
* [Change-Point Detection using Singular Spectrum Transformation (SST)](anomaly/sst.md)
* [ChangeFinder: Detecting Outlier and Change-Point Simultaneously](anomaly/changefinder.md)

## Part XI - Clustering

* [Latent Dirichlet Allocation](clustering/lda.md)
* [Probabilistic Latent Semantic Analysis](clustering/plsa.md)

## Part XII - GeoSpatial Functions

* [Lat/Lon functions](geospatial/latlon.md)

## Part XIII - Hivemall on Spark

* [Getting Started](spark/getting_started/README.md)
    * [Installation](spark/getting_started/installation.md)

* [Binary Classification](spark/binaryclass/index.md)
    * [a9a tutorial for DataFrame](spark/binaryclass/a9a_df.md)
    * [a9a tutorial for SQL](spark/binaryclass/a9a_sql.md)

* [Regression](spark/binaryclass/index.md)
    * [E2006-tfidf regression tutorial for DataFrame](spark/regression/e2006_df.md)
    * [E2006-tfidf regression tutorial for SQL](spark/regression/e2006_sql.md)

* [Generic features](spark/misc/misc.md)
    * [Top-k join processing](spark/misc/topk_join.md)
    * [Other utility functions](spark/misc/functions.md)

## Part XIV - Hivemall on Docker

* [Getting Started](docker/getting_started.md)

## Part XIV - External References

* [Hivemall on Apache Pig](https://github.com/daijyc/hivemall/wiki/PigHome)

