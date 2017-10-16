/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.hive

import java.util.UUID

import org.apache.spark.annotation.Experimental
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.HivemallFeature
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, VectorUDT}
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Generate, JoinTopK, LogicalPlan}
import org.apache.spark.sql.execution.UserProvidedPlanner
import org.apache.spark.sql.execution.datasources.csv.{CsvToStruct, StructToCsv}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String


/**
 * Hivemall wrapper and some utility functions for DataFrame. These functions below derives
 * from `resources/ddl/define-all-as-permanent.hive`.
 *
 * @groupname regression
 * @groupname classifier
 * @groupname classifier.multiclass
 * @groupname recommend
 * @groupname topicmodel
 * @groupname geospatial
 * @groupname smile
 * @groupname xgboost
 * @groupname anomaly
 * @groupname knn.similarity
 * @groupname knn.distance
 * @groupname knn.lsh
 * @groupname ftvec
 * @groupname ftvec.amplify
 * @groupname ftvec.hashing
 * @groupname ftvec.paring
 * @groupname ftvec.scaling
 * @groupname ftvec.selection
 * @groupname ftvec.conv
 * @groupname ftvec.trans
 * @groupname ftvec.ranking
 * @groupname tools
 * @groupname tools.array
 * @groupname tools.bits
 * @groupname tools.compress
 * @groupname tools.map
 * @groupname tools.text
 * @groupname misc
 *
 * A list of unsupported functions is as follows:
 *  * smile
 *   - guess_attribute_types
 *  * mapred functions
 *   - taskid
 *   - jobid
 *   - rownum
 *   - distcache_gets
 *   - jobconf_gets
 *  * matrix factorization
 *   - mf_predict
 *   - train_mf_sgd
 *   - train_mf_adagrad
 *   - train_bprmf
 *   - bprmf_predict
 *  * Factorization Machine
 *   - fm_predict
 *   - train_fm
 *   - train_ffm
 *   - ffm_predict
 */
final class HivemallOps(df: DataFrame) extends Logging {
  import internal.HivemallOpsImpl._

  private lazy val _sparkSession = df.sparkSession
  private lazy val _strategy = new UserProvidedPlanner(_sparkSession.sqlContext.conf)

  /**
   * @see [[hivemall.regression.GeneralRegressorUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_regressor(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.GeneralRegressorUDTF",
      "train_regressor",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.AdaDeltaUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_adadelta_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.AdaDeltaUDTF",
      "train_adadelta_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.AdaGradUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_adagrad_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.AdaGradUDTF",
      "train_adagrad_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.AROWRegressionUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_arow_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.AROWRegressionUDTF",
      "train_arow_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.regression.AROWRegressionUDTF.AROWe]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_arowe_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.AROWRegressionUDTF$AROWe",
      "train_arowe_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.regression.AROWRegressionUDTF.AROWe2]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_arowe2_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.AROWRegressionUDTF$AROWe2",
      "train_arowe2_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.regression.LogressUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_logistic_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.LogressUDTF",
      "train_logistic_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.PassiveAggressiveRegressionUDTF]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_pa1_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.PassiveAggressiveRegressionUDTF",
      "train_pa1_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.PassiveAggressiveRegressionUDTF.PA1a]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_pa1a_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.PassiveAggressiveRegressionUDTF$PA1a",
      "train_pa1a_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.PassiveAggressiveRegressionUDTF.PA2]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_pa2_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.PassiveAggressiveRegressionUDTF$PA2",
      "train_pa2_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.regression.PassiveAggressiveRegressionUDTF.PA2a]]
   * @group regression
   */
  @scala.annotation.varargs
  def train_pa2a_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.regression.PassiveAggressiveRegressionUDTF$PA2a",
      "train_pa2a_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.GeneralClassifierUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_classifier(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.GeneralClassifierUDTF",
      "train_classifier",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.PerceptronUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_perceptron(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.PerceptronUDTF",
      "train_perceptron",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.PassiveAggressiveUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_pa(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.PassiveAggressiveUDTF",
      "train_pa",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.PassiveAggressiveUDTF.PA1]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_pa1(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.PassiveAggressiveUDTF$PA1",
      "train_pa1",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.PassiveAggressiveUDTF.PA2]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_pa2(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.PassiveAggressiveUDTF$PA2",
      "train_pa2",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.ConfidenceWeightedUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_cw(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.ConfidenceWeightedUDTF",
      "train_cw",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.AROWClassifierUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_arow(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.AROWClassifierUDTF",
      "train_arow",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.AROWClassifierUDTF.AROWh]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_arowh(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.AROWClassifierUDTF$AROWh",
      "train_arowh",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.SoftConfideceWeightedUDTF.SCW1]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_scw(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.SoftConfideceWeightedUDTF$SCW1",
      "train_scw",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.SoftConfideceWeightedUDTF.SCW1]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_scw2(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.SoftConfideceWeightedUDTF$SCW2",
      "train_scw2",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.AdaGradRDAUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_adagrad_rda(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.AdaGradRDAUDTF",
      "train_adagrad_rda",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.KernelExpansionPassiveAggressiveUDTF]]
   * @group classifier
   */
  @scala.annotation.varargs
  def train_kpa(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.KernelExpansionPassiveAggressiveUDTF",
      "train_kpa",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("h", "hk", "w0", "w1", "w2", "w3")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassPerceptronUDTF]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_perceptron(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassPerceptronUDTF",
      "train_multiclass_perceptron",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_pa(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF",
      "train_multiclass_pa",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF.PA1]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_pa1(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF$PA1",
      "train_multiclass_pa1",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF.PA2]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_pa2(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassPassiveAggressiveUDTF$PA2",
      "train_multiclass_pa2",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassConfidenceWeightedUDTF]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_cw(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassConfidenceWeightedUDTF",
      "train_multiclass_cw",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassAROWClassifierUDTF]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_arow(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassAROWClassifierUDTF",
      "train_multiclass_arow",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassAROWClassifierUDTF.AROWh]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_arowh(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassAROWClassifierUDTF$AROWh",
      "train_multiclass_arowh",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassSoftConfidenceWeightedUDTF.SCW1]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_scw(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassSoftConfidenceWeightedUDTF$SCW1",
      "train_multiclass_scw",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.classifier.multiclass.MulticlassSoftConfidenceWeightedUDTF.SCW2]]
   * @group classifier.multiclass
   */
  @scala.annotation.varargs
  def train_multiclass_scw2(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.classifier.multiclass.MulticlassSoftConfidenceWeightedUDTF$SCW2",
      "train_multiclass_scw2",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("label", "feature", "weight", "conv")
    )
  }

  /**
   * @see [[hivemall.recommend.SlimUDTF]]
   * @group recommend
   */
  @scala.annotation.varargs
  def train_slim(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.recommend.SlimUDTF",
      "train_slim",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("j", "nn", "w")
    )
  }

  /**
   * @see [[hivemall.topicmodel.LDAUDTF]]
   * @group topicmodel
   */
  @scala.annotation.varargs
  def train_lda(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.topicmodel.LDAUDTF",
      "train_lda",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("topic", "word", "score")
    )
  }

  /**
   * @see [[hivemall.topicmodel.PLSAUDTF]]
   * @group topicmodel
   */
  @scala.annotation.varargs
  def train_plsa(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.topicmodel.PLSAUDTF",
      "train_plsa",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("topic", "word", "score")
    )
  }

  /**
   * @see [[hivemall.smile.regression.RandomForestRegressionUDTF]]
   * @group smile
   */
  @scala.annotation.varargs
  def train_randomforest_regressor(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.smile.regression.RandomForestRegressionUDTF",
      "train_randomforest_regressor",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("model_id", "model_type", "pred_model", "var_importance", "oob_errors", "oob_tests")
    )
  }

  /**
   * @see [[hivemall.smile.classification.RandomForestClassifierUDTF]]
   * @group smile
   */
  @scala.annotation.varargs
  def train_randomforest_classifier(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.smile.classification.RandomForestClassifierUDTF",
      "train_randomforest_classifier",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("model_id", "model_type", "pred_model", "var_importance", "oob_errors", "oob_tests")
    )
  }

  /**
   * :: Experimental ::
   * @see [[hivemall.xgboost.regression.XGBoostRegressionUDTF]]
   * @group xgboost
   */
  @Experimental
  @scala.annotation.varargs
  def train_xgboost_regr(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.xgboost.regression.XGBoostRegressionUDTF",
      "train_xgboost_regr",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("model_id", "pred_model")
    )
  }

  /**
   * :: Experimental ::
   * @see [[hivemall.xgboost.classification.XGBoostBinaryClassifierUDTF]]
   * @group xgboost
   */
  @Experimental
  @scala.annotation.varargs
  def train_xgboost_classifier(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.xgboost.classification.XGBoostBinaryClassifierUDTF",
      "train_xgboost_classifier",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("model_id", "pred_model")
    )
  }

  /**
   * :: Experimental ::
   * @see [[hivemall.xgboost.classification.XGBoostMulticlassClassifierUDTF]]
   * @group xgboost
   */
  @Experimental
  @scala.annotation.varargs
  def train_xgboost_multiclass_classifier(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.xgboost.classification.XGBoostMulticlassClassifierUDTF",
      "train_xgboost_multiclass_classifier",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("model_id", "pred_model")
    )
  }

  /**
   * :: Experimental ::
   * @see [[hivemall.xgboost.tools.XGBoostPredictUDTF]]
   * @group xgboost
   */
  @Experimental
  @scala.annotation.varargs
  def xgboost_predict(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.xgboost.tools.XGBoostPredictUDTF",
      "xgboost_predict",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("rowid", "predicted")
    )
  }

  /**
   * :: Experimental ::
   * @see [[hivemall.xgboost.tools.XGBoostMulticlassPredictUDTF]]
   * @group xgboost
   */
  @Experimental
  @scala.annotation.varargs
  def xgboost_multiclass_predict(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.xgboost.tools.XGBoostMulticlassPredictUDTF",
      "xgboost_multiclass_predict",
      setMixServs(toHivemallFeatures(exprs)),
      Seq("rowid", "label", "probability")
    )
  }

  /**
   * @see [[hivemall.knn.similarity.DIMSUMMapperUDTF]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def dimsum_mapper(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.knn.similarity.DIMSUMMapperUDTF",
      "dimsum_mapper",
      exprs,
      Seq("j", "k", "b_jk")
    )
  }

  /**
   * @see [[hivemall.knn.lsh.MinHashUDTF]]
   * @group knn.lsh
   */
  @scala.annotation.varargs
  def minhash(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.knn.lsh.MinHashUDTF",
      "minhash",
      exprs,
      Seq("clusterid", "item")
    )
  }

  /**
   * @see [[hivemall.ftvec.amplify.AmplifierUDTF]]
   * @group ftvec.amplify
   */
  @scala.annotation.varargs
  def amplify(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.amplify.AmplifierUDTF",
      "amplify",
      exprs,
      Seq("clusterid", "item")
    )
  }

  /**
   * @see [[hivemall.ftvec.amplify.RandomAmplifierUDTF]]
   * @group ftvec.amplify
   */
  @scala.annotation.varargs
  def rand_amplify(exprs: Column*): DataFrame = withTypedPlan {
    throw new UnsupportedOperationException("`rand_amplify` not supported yet")
  }

  /**
   * Amplifies and shuffle data inside partitions.
   * @group ftvec.amplify
   */
  def part_amplify(xtimes: Column): DataFrame = {
    val xtimesInt = xtimes.expr match {
      case Literal(v: Any, IntegerType) => v.asInstanceOf[Int]
      case e => throw new AnalysisException("`xtimes` must be integer, however " + e)
    }
    val rdd = df.rdd.mapPartitions({ iter =>
      val elems = iter.flatMap{ row =>
        Seq.fill[Row](xtimesInt)(row)
      }
      // Need to check how this shuffling affects results
      scala.util.Random.shuffle(elems)
    }, true)
    df.sqlContext.createDataFrame(rdd, df.schema)
  }

  /**
   * Quantifies input columns.
   * @see [[hivemall.ftvec.conv.QuantifyColumnsUDTF]]
   * @group ftvec.conv
   */
  @scala.annotation.varargs
  def quantify(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.conv.QuantifyColumnsUDTF",
      "quantify",
      exprs,
      (0 until exprs.size - 1).map(i => s"c$i")
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.BinarizeLabelUDTF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def binarize_label(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.trans.BinarizeLabelUDTF",
      "binarize_label",
      exprs,
      (0 until exprs.size - 1).map(i => s"c$i")
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.QuantifiedFeaturesUDTF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def quantified_features(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.trans.QuantifiedFeaturesUDTF",
      "quantified_features",
      exprs,
      Seq("features")
    )
  }

  /**
   * @see [[hivemall.ftvec.ranking.BprSamplingUDTF]]
   * @group ftvec.ranking
   */
  @scala.annotation.varargs
  def bpr_sampling(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.ranking.BprSamplingUDTF",
      "bpr_sampling",
      exprs,
      Seq("user", "pos_item", "neg_item")
    )
  }

  /**
   * @see [[hivemall.ftvec.ranking.ItemPairsSamplingUDTF]]
   * @group ftvec.ranking
   */
  @scala.annotation.varargs
  def item_pairs_sampling(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.ranking.ItemPairsSamplingUDTF",
      "item_pairs_sampling",
      exprs,
      Seq("pos_item_id", "neg_item_id")
    )
  }

  /**
   * @see [[hivemall.ftvec.ranking.PopulateNotInUDTF]]
   * @group ftvec.ranking
   */
  @scala.annotation.varargs
  def populate_not_in(exprs: Column*): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.ftvec.ranking.PopulateNotInUDTF",
      "populate_not_in",
      exprs,
      Seq("item")
    )
  }

  /**
   * Splits Seq[String] into pieces.
   * @group ftvec
   */
  def explode_array(features: Column): DataFrame = {
    df.explode(features) { case Row(v: Seq[_]) =>
      // Type erasure removes the component type in Seq
      v.map(s => HivemallFeature(s.asInstanceOf[String]))
    }
  }

  /**
   * Splits [[Vector]] into pieces.
   * @group ftvec
   */
  def explode_vector(features: Column): DataFrame = {
    val elementSchema = StructType(
      StructField("feature", StringType) :: StructField("weight", DoubleType) :: Nil)
    val explodeFunc: Row => TraversableOnce[InternalRow] = (row: Row) => {
      row.get(0) match {
        case dv: DenseVector =>
          dv.values.zipWithIndex.map {
            case (value, index) =>
              InternalRow(UTF8String.fromString(s"$index"), value)
          }
        case sv: SparseVector =>
          sv.values.zip(sv.indices).map {
            case (value, index) =>
              InternalRow(UTF8String.fromString(s"$index"), value)
          }
      }
    }
    withTypedPlan {
      Generate(
        UserDefinedGenerator(elementSchema, explodeFunc, features.expr :: Nil),
        join = true, outer = false, None,
        generatorOutput = Nil,
        df.logicalPlan)
    }
  }

  /**
   * @see [[hivemall.tools.GenerateSeriesUDTF]]
   * @group tools
   */
  def generate_series(start: Column, end: Column): DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.tools.GenerateSeriesUDTF",
      "generate_series",
      start :: end :: Nil,
      Seq("generate_series")
    )
  }

  /**
   * Returns `top-k` records for each `group`.
   * @group misc
   */
  def each_top_k(k: Column, score: Column, group: Column*): DataFrame = withTypedPlan {
    val kInt = k.expr match {
      case Literal(v: Any, IntegerType) => v.asInstanceOf[Int]
      case e => throw new AnalysisException("`k` must be integer, however " + e)
    }
    if (kInt == 0) {
      throw new AnalysisException("`k` must not have 0")
    }
    val clusterDf = df.repartition(group: _*).sortWithinPartitions(group: _*)
      .select(score, Column("*"))
    val analyzedPlan = clusterDf.queryExecution.analyzed
    val inputAttrs = analyzedPlan.output
    val scoreExpr = BindReferences.bindReference(analyzedPlan.expressions.head, inputAttrs)
    val groupNames = group.map { _.expr match {
      case ne: NamedExpression => ne.name
      case ua: UnresolvedAttribute => ua.name
    }}
    val groupExprs = analyzedPlan.expressions.filter {
      case ne: NamedExpression => groupNames.contains(ne.name)
    }.map { e =>
      BindReferences.bindReference(e, inputAttrs)
    }
    val rankField = StructField("rank", IntegerType)
    Generate(
      generator = EachTopK(
        k = kInt,
        scoreExpr = scoreExpr,
        groupExprs = groupExprs,
        elementSchema = StructType(
          rankField +: inputAttrs.map(d => StructField(d.name, d.dataType))
        ),
        children = inputAttrs
      ),
      join = false,
      outer = false,
      qualifier = None,
      generatorOutput = Seq(rankField.name).map(UnresolvedAttribute(_)) ++ inputAttrs,
      child = analyzedPlan
    )
  }

  /**
   * :: Experimental ::
   * Joins input two tables with the given keys and the top-k highest `score` values.
   * @group misc
   */
  @Experimental
  def top_k_join(k: Column, right: DataFrame, joinExprs: Column, score: Column)
    : DataFrame = withTypedPlanInCustomStrategy {
    val kInt = k.expr match {
      case Literal(v: Any, IntegerType) => v.asInstanceOf[Int]
      case e => throw new AnalysisException("`k` must be integer, however " + e)
    }
    if (kInt == 0) {
      throw new AnalysisException("`k` must not have 0")
    }
    JoinTopK(kInt, df.logicalPlan, right.logicalPlan, Inner, Option(joinExprs.expr))(score.named)
  }

  private def doFlatten(schema: StructType, separator: Char, prefixParts: Seq[String] = Seq.empty)
    : Seq[Column] = {
    schema.fields.flatMap { f =>
      val colNameParts = prefixParts :+ f.name
      f.dataType match {
        case st: StructType =>
          doFlatten(st, separator, colNameParts)
        case _ =>
          col(colNameParts.mkString(".")).as(colNameParts.mkString(separator.toString)) :: Nil
      }
    }
  }

  // Converts string representation of a character to actual character
  @throws[IllegalArgumentException]
  private def toChar(str: String): Char = {
    if (str.length == 1) {
      str.charAt(0) match {
        case '$' | '_' | '.' => str.charAt(0)
        case _ => throw new IllegalArgumentException(
          "Must use '$', '_', or '.' for separator, but got " + str)
      }
    } else {
      throw new IllegalArgumentException(
        s"Separator cannot be more than one character: $str")
    }
  }

  /**
   * Flattens a nested schema into a flat one.
   * @group misc
   *
   * For example:
   * {{{
   *  scala> val df = Seq((0, (1, (3.0, "a")), (5, 0.9))).toDF()
   *  scala> df.printSchema
   *  root
   *   |-- _1: integer (nullable = false)
   *   |-- _2: struct (nullable = true)
   *   |    |-- _1: integer (nullable = false)
   *   |    |-- _2: struct (nullable = true)
   *   |    |    |-- _1: double (nullable = false)
   *   |    |    |-- _2: string (nullable = true)
   *   |-- _3: struct (nullable = true)
   *   |    |-- _1: integer (nullable = false)
   *   |    |-- _2: double (nullable = false)
   *
   *  scala> df.flatten(separator = "$").printSchema
   *  root
   *   |-- _1: integer (nullable = false)
   *   |-- _2$_1: integer (nullable = true)
   *   |-- _2$_2$_1: double (nullable = true)
   *   |-- _2$_2$_2: string (nullable = true)
   *   |-- _3$_1: integer (nullable = true)
   *   |-- _3$_2: double (nullable = true)
   * }}}
   */
  def flatten(separator: String = "$"): DataFrame =
    df.select(doFlatten(df.schema, toChar(separator)): _*)

  /**
   * @see [[hivemall.dataset.LogisticRegressionDataGeneratorUDTF]]
   * @group misc
   */
  @scala.annotation.varargs
  def lr_datagen(exprs: Column*): Dataset[Row] = withTypedPlan {
    planHiveGenericUDTF(
      df,
      "hivemall.dataset.LogisticRegressionDataGeneratorUDTFWrapper",
      "lr_datagen",
      exprs,
      Seq("label", "features")
    )
  }

  /**
   * Returns all the columns as Seq[Column] in this [[DataFrame]].
   */
  private[sql] def cols: Seq[Column] = {
    df.schema.fields.map(col => df.col(col.name)).toSeq
  }

  /**
   * :: Experimental ::
   * If a parameter '-mix' does not exist in a 3rd argument,
   * set it from an environmental variable
   * 'HIVEMALL_MIX_SERVERS'.
   *
   * TODO: This could work if '--deploy-mode' has 'client';
   * otherwise, we need to set HIVEMALL_MIX_SERVERS
   * in all possible spark workers.
   */
  @Experimental
  private def setMixServs(exprs: Seq[Column]): Seq[Column] = {
    val mixes = System.getenv("HIVEMALL_MIX_SERVERS")
    if (mixes != null && !mixes.isEmpty()) {
      val groupId = df.sqlContext.sparkContext.applicationId + "-" + UUID.randomUUID
      logInfo(s"set '${mixes}' as default mix servers (session: ${groupId})")
      exprs.size match {
        case 2 => exprs :+ Column(
          Literal.create(s"-mix ${mixes} -mix_session ${groupId}", StringType))
        /** TODO: Add codes in the case where exprs.size == 3. */
        case _ => exprs
      }
    } else {
      exprs
    }
  }

  /**
   * If the input is a [[Vector]], transform it into Hivemall features.
   */
  @inline private def toHivemallFeatures(exprs: Seq[Column]): Seq[Column] = {
    df.select(exprs: _*).queryExecution.analyzed.schema.zip(exprs).map {
      case (StructField(_, _: VectorUDT, _, _), c) => HivemallUtils.to_hivemall_features(c)
      case (_, c) => c
    }
  }

  /**
   * A convenient function to wrap a logical plan and produce a DataFrame.
   */
  @inline private def withTypedPlan(logicalPlan: => LogicalPlan): DataFrame = {
    val queryExecution = _sparkSession.sessionState.executePlan(logicalPlan)
    val outputSchema = queryExecution.sparkPlan.schema
    new Dataset[Row](df.sparkSession, queryExecution, RowEncoder(outputSchema))
  }

  @inline private def withTypedPlanInCustomStrategy(logicalPlan: => LogicalPlan)
    : DataFrame = {
    // Inject custom strategies
    if (!_sparkSession.experimental.extraStrategies.contains(_strategy)) {
      _sparkSession.experimental.extraStrategies = Seq(_strategy)
    }
    withTypedPlan(logicalPlan)
  }
}

object HivemallOps {
  import internal.HivemallOpsImpl._

  /**
   * Implicitly inject the [[HivemallOps]] into [[DataFrame]].
   */
  implicit def dataFrameToHivemallOps(df: DataFrame): HivemallOps =
    new HivemallOps(df)

  /**
   * @see [[hivemall.HivemallVersionUDF]]
   * @group misc
   */
  def hivemall_version(): Column = withExpr {
    planHiveUDF(
      "hivemall.HivemallVersionUDF",
      "hivemall_version",
      Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.TileUDF]]
   * @group geospatial
   */
  def tile(lat: Column, lon: Column, zoom: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.TileUDF",
      "tile",
      lat :: lon :: zoom :: Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.MapURLUDF]]
   * @group geospatial
   */
  @scala.annotation.varargs
  def map_url(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.MapURLUDF",
      "map_url",
      exprs
    )
  }

  /**
   * @see [[hivemall.geospatial.Lat2TileYUDF]]
   * @group geospatial
   */
  def lat2tiley(lat: Column, zoom: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.Lat2TileYUDF",
      "lat2tiley",
      lat :: zoom :: Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.Lon2TileXUDF]]
   * @group geospatial
   */
  def lon2tilex(lon: Column, zoom: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.Lon2TileXUDF",
      "lon2tilex",
      lon :: zoom :: Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.TileX2LonUDF]]
   * @group geospatial
   */
  def tilex2lon(x: Column, zoom: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.TileX2LonUDF",
      "tilex2lon",
      x :: zoom :: Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.TileY2LatUDF]]
   * @group geospatial
   */
  def tiley2lat(y: Column, zoom: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.TileY2LatUDF",
      "tiley2lat",
      y :: zoom :: Nil
    )
  }

  /**
   * @see [[hivemall.geospatial.HaversineDistanceUDF]]
   * @group geospatial
   */
  @scala.annotation.varargs
  def haversine_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.geospatial.HaversineDistanceUDF",
      "haversine_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.smile.tools.TreePredictUDF]]
   * @group smile
   */
  @scala.annotation.varargs
  def tree_predict(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.smile.tools.TreePredictUDF",
      "tree_predict",
      exprs
    )
  }

  /**
   * @see [[hivemall.smile.tools.TreeExportUDF]]
   * @group smile
   */
  @scala.annotation.varargs
  def tree_export(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.smile.tools.TreeExportUDF",
      "tree_export",
      exprs
    )
  }

  /**
   * @see [[hivemall.anomaly.ChangeFinderUDF]]
   * @group anomaly
   */
  @scala.annotation.varargs
  def changefinder(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.anomaly.ChangeFinderUDF",
      "changefinder",
      exprs
    )
  }

  /**
   * @see [[hivemall.anomaly.SingularSpectrumTransformUDF]]
   * @group anomaly
   */
  @scala.annotation.varargs
  def sst(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.anomaly.SingularSpectrumTransformUDF",
      "sst",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.similarity.CosineSimilarityUDF]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def cosine_similarity(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.similarity.CosineSimilarityUDF",
      "cosine_similarity",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.similarity.JaccardIndexUDF]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def jaccard_similarity(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.similarity.JaccardIndexUDF",
      "jaccard_similarity",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.similarity.AngularSimilarityUDF]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def angular_similarity(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.similarity.AngularSimilarityUDF",
      "angular_similarity",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.similarity.EuclidSimilarity]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def euclid_similarity(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.similarity.EuclidSimilarity",
      "euclid_similarity",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.similarity.Distance2SimilarityUDF]]
   * @group knn.similarity
   */
  @scala.annotation.varargs
  def distance2similarity(exprs: Column*): Column = withExpr {
    // TODO: Need a wrapper class because of using unsupported types
    planHiveGenericUDF(
      "hivemall.knn.similarity.Distance2SimilarityUDF",
      "distance2similarity",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.HammingDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def hamming_distance(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.distance.HammingDistanceUDF",
      "hamming_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.PopcountUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def popcnt(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.distance.PopcountUDF",
      "popcnt",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.KLDivergenceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def kld(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.distance.KLDivergenceUDF",
      "kld",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.EuclidDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def euclid_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.distance.EuclidDistanceUDF",
      "euclid_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.CosineDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def cosine_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.distance.CosineDistanceUDF",
      "cosine_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.AngularDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def angular_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.distance.AngularDistanceUDF",
      "angular_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.JaccardDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def jaccard_distance(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.distance.JaccardDistanceUDF",
      "jaccard_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.ManhattanDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def manhattan_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.distance.ManhattanDistanceUDF",
      "manhattan_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.distance.MinkowskiDistanceUDF]]
   * @group knn.distance
   */
  @scala.annotation.varargs
  def minkowski_distance(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.distance.MinkowskiDistanceUDF",
      "minkowski_distance",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.lsh.bBitMinHashUDF]]
   * @group knn.lsh
   */
  @scala.annotation.varargs
  def bbit_minhash(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.knn.lsh.bBitMinHashUDF",
      "bbit_minhash",
      exprs
    )
  }

  /**
   * @see [[hivemall.knn.lsh.MinHashesUDFWrapper]]
   * @group knn.lsh
   */
  @scala.annotation.varargs
  def minhashes(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.knn.lsh.MinHashesUDFWrapper",
      "minhashes",
      exprs
    )
  }

  /**
   * Returns new features with `1.0` (bias) appended to the input features.
   * @see [[hivemall.ftvec.AddBiasUDFWrapper]]
   * @group ftvec
   */
  def add_bias(expr: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.AddBiasUDFWrapper",
      "add_bias",
      expr :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.ExtractFeatureUDFWrapper]]
   * @group ftvec
   *
   * TODO: This throws java.lang.ClassCastException because
   * HiveInspectors.toInspector has a bug in spark.
   * Need to fix it later.
   */
  def extract_feature(expr: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.ExtractFeatureUDFWrapper",
      "extract_feature",
      expr :: Nil
    )
  }.as("feature")

  /**
   * @see [[hivemall.ftvec.ExtractWeightUDFWrapper]]
   * @group ftvec
   *
   * TODO: This throws java.lang.ClassCastException because
   * HiveInspectors.toInspector has a bug in spark.
   * Need to fix it later.
   */
  def extract_weight(expr: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.ExtractWeightUDFWrapper",
      "extract_weight",
      expr :: Nil
    )
  }.as("value")

  /**
   * @see [[hivemall.ftvec.AddFeatureIndexUDFWrapper]]
   * @group ftvec
   */
  def add_feature_index(features: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.AddFeatureIndexUDFWrapper",
      "add_feature_index",
      features :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.SortByFeatureUDFWrapper]]
   * @group ftvec
   */
  def sort_by_feature(expr: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.SortByFeatureUDFWrapper",
      "sort_by_feature",
      expr :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.hashing.MurmurHash3UDF]]
   * @group ftvec.hashing
   */
  def mhash(expr: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.hashing.MurmurHash3UDF",
      "mhash",
      expr :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.hashing.Sha1UDF]]
   * @group ftvec.hashing
   */
  @scala.annotation.varargs
  def sha1(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.hashing.Sha1UDF",
      "sha1",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.hashing.ArrayHashValuesUDF]]
   * @group ftvec.hashing
   */
  @scala.annotation.varargs
  def array_hash_values(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.hashing.ArrayHashValuesUDF",
      "array_hash_values",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.hashing.ArrayPrefixedHashValuesUDF]]
   * @group ftvec.hashing
   */
  @scala.annotation.varargs
  def prefixed_hash_values(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.hashing.ArrayPrefixedHashValuesUDF",
      "prefixed_hash_values",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.hashing.FeatureHashingUDF]]
   * @group ftvec.hashing
   */
  @scala.annotation.varargs
  def feature_hashing(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.hashing.FeatureHashingUDF",
      "feature_hashing",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.pairing.PolynomialFeaturesUDF]]
   * @group ftvec.paring
   */
  @scala.annotation.varargs
  def polynomial_features(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.pairing.PolynomialFeaturesUDF",
      "polynomial_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.pairing.PoweredFeaturesUDF]]
   * @group ftvec.paring
   */
  @scala.annotation.varargs
  def powered_features(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.pairing.PoweredFeaturesUDF",
      "powered_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.scaling.RescaleUDF]]
   * @group ftvec.scaling
   */
  def rescale(value: Column, max: Column, min: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.scaling.RescaleUDF",
      "rescale",
      value.cast(FloatType) :: max :: min :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.scaling.ZScoreUDF]]
   * @group ftvec.scaling
   */
  @scala.annotation.varargs
  def zscore(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.scaling.ZScoreUDF",
      "zscore",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.scaling.L2NormalizationUDFWrapper]]
   * @group ftvec.scaling
   */
  def l2_normalize(expr: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.scaling.L2NormalizationUDFWrapper",
      "normalize",
      expr :: Nil
    )
  }

  /**
   * @see [[hivemall.ftvec.selection.ChiSquareUDF]]
   * @group ftvec.selection
   */
  def chi2(observed: Column, expected: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.selection.ChiSquareUDF",
      "chi2",
      Seq(observed, expected)
    )
  }

  /**
   * @see [[hivemall.ftvec.conv.ToDenseFeaturesUDF]]
   * @group ftvec.conv
   */
  @scala.annotation.varargs
  def to_dense_features(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.conv.ToDenseFeaturesUDF",
      "to_dense_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.conv.ToSparseFeaturesUDF]]
   * @group ftvec.conv
   */
  @scala.annotation.varargs
  def to_sparse_features(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.ftvec.conv.ToSparseFeaturesUDF",
      "to_sparse_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.binning.FeatureBinningUDF]]
   * @group ftvec.conv
   */
  @scala.annotation.varargs
  def feature_binning(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.binning.FeatureBinningUDF",
      "feature_binning",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.VectorizeFeaturesUDF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def vectorize_features(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.VectorizeFeaturesUDF",
      "vectorize_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.CategoricalFeaturesUDF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def categorical_features(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.CategoricalFeaturesUDF",
      "categorical_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.FFMFeaturesUDF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def ffm_features(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.FFMFeaturesUDF",
      "ffm_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.IndexedFeatures]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def indexed_features(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.IndexedFeatures",
      "indexed_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.QuantitativeFeaturesUDF]]
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def quantitative_features(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.QuantitativeFeaturesUDF",
      "quantitative_features",
      exprs
    )
  }

  /**
   * @see [[hivemall.ftvec.trans.AddFieldIndicesUDF]]
   * @group ftvec.trans
   */
  def add_field_indices(features: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.ftvec.trans.AddFieldIndicesUDF",
      "add_field_indices",
      features :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.ConvertLabelUDF]]
   * @group tools
   */
  def convert_label(label: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.ConvertLabelUDF",
      "convert_label",
      label :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.RankSequenceUDF]]
   * @group tools
   */
  def x_rank(key: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.RankSequenceUDF",
      "x_rank",
      key :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.AllocFloatArrayUDF]]
   * @group tools.array
   */
  def float_array(nDims: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.AllocFloatArrayUDF",
      "float_array",
      nDims :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.ArrayRemoveUDF]]
   * @group tools.array
   */
  def array_remove(original: Column, target: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.ArrayRemoveUDF",
      "array_remove",
      original :: target :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.SortAndUniqArrayUDF]]
   * @group tools.array
   */
  def sort_and_uniq_array(ar: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.SortAndUniqArrayUDF",
      "sort_and_uniq_array",
      ar :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.SubarrayEndWithUDF]]
   * @group tools.array
   */
  def subarray_endwith(original: Column, key: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.SubarrayEndWithUDF",
      "subarray_endwith",
      original :: key :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.ArrayConcatUDF]]
   * @group tools.array
   */
  @scala.annotation.varargs
  def array_concat(arrays: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.array.ArrayConcatUDF",
      "array_concat",
      arrays
    )
  }

  /**
   * @see [[hivemall.tools.array.SubarrayUDF]]
   * @group tools.array
   */
  def subarray(original: Column, fromIndex: Column, toIndex: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.SubarrayUDF",
      "subarray",
      original :: fromIndex :: toIndex :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.ToStringArrayUDF]]
   * @group tools.array
   */
  def to_string_array(ar: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.array.ToStringArrayUDF",
      "to_string_array",
      ar :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.array.ArrayIntersectUDF]]
   * @group tools.array
   */
  @scala.annotation.varargs
  def array_intersect(arrays: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.array.ArrayIntersectUDF",
      "array_intersect",
      arrays
    )
  }

  /**
   * @see [[hivemall.tools.array.SelectKBestUDF]]
   * @group tools.array
   */
  def select_k_best(X: Column, importanceList: Column, k: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.array.SelectKBestUDF",
      "select_k_best",
      Seq(X, importanceList, k)
    )
  }

  /**
   * @see [[hivemall.tools.bits.ToBitsUDF]]
   * @group tools.bits
   */
  def to_bits(indexes: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.bits.ToBitsUDF",
      "to_bits",
      indexes :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.bits.UnBitsUDF]]
   * @group tools.bits
   */
  def unbits(bitset: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.bits.UnBitsUDF",
      "unbits",
      bitset :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.bits.BitsORUDF]]
   * @group tools.bits
   */
  @scala.annotation.varargs
  def bits_or(bits: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.bits.BitsORUDF",
      "bits_or",
      bits
    )
  }

  /**
   * @see [[hivemall.tools.compress.InflateUDF]]
   * @group tools.compress
   */
  @scala.annotation.varargs
  def inflate(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.compress.InflateUDF",
      "inflate",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.compress.DeflateUDF]]
   * @group tools.compress
   */
  @scala.annotation.varargs
  def deflate(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.compress.DeflateUDF",
      "deflate",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.map.MapGetSumUDF]]
   * @group tools.map
   */
  @scala.annotation.varargs
  def map_get_sum(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.map.MapGetSumUDF",
      "map_get_sum",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.map.MapTailNUDF]]
   * @group tools.map
   */
  @scala.annotation.varargs
  def map_tail_n(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.map.MapTailNUDF",
      "map_tail_n",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.text.TokenizeUDF]]
   * @group tools.text
   */
  @scala.annotation.varargs
  def tokenize(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.TokenizeUDF",
      "tokenize",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.text.StopwordUDF]]
   * @group tools.text
   */
  def is_stopword(word: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.StopwordUDF",
      "is_stopword",
      word :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.text.SingularizeUDF]]
   * @group tools.text
   */
  def singularize(word: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.SingularizeUDF",
      "singularize",
      word :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.text.SplitWordsUDF]]
   * @group tools.text
   */
  @scala.annotation.varargs
  def split_words(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.SplitWordsUDF",
      "split_words",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.text.NormalizeUnicodeUDF]]
   * @group tools.text
   */
  @scala.annotation.varargs
  def normalize_unicode(exprs: Column*): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.NormalizeUnicodeUDF",
      "normalize_unicode",
      exprs
    )
  }

  /**
   * @see [[hivemall.tools.text.Base91UDF]]
   * @group tools.text
   */
  def base91(bin: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.text.Base91UDF",
      "base91",
      bin :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.text.Unbase91UDF]]
   * @group tools.text
   */
  def unbase91(base91String: Column): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.text.Unbase91UDF",
      "unbase91",
      base91String :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.text.WordNgramsUDF]]
   * @group tools.text
   */
  def word_ngrams(words: Column, minSize: Column, maxSize: Column): Column = withExpr {
    planHiveUDF(
      "hivemall.tools.text.WordNgramsUDF",
      "word_ngrams",
      words :: minSize :: maxSize :: Nil
    )
  }

  /**
   * @see [[hivemall.tools.math.SigmoidGenericUDF]]
   * @group misc
   */
  def sigmoid(expr: Column): Column = {
    val one: () => Literal = () => Literal.create(1.0, DoubleType)
    Column(one()) / (Column(one()) + exp(-expr))
  }

  /**
   * @see [[hivemall.tools.mapred.RowIdUDFWrapper]]
   * @group misc
   */
  def rowid(): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.mapred.RowIdUDFWrapper",
      "rowid",
       Nil
    )
  }.as("rowid")

  /**
   * Parses a column containing a CSV string into a [[StructType]] with the specified schema.
   * Returns `null`, in the case of an unparseable string.
   * @group misc
   *
   * @param e a string column containing CSV data.
   * @param schema the schema to use when parsing the csv string
   * @param options options to control how the csv is parsed. accepts the same options and the
   *                csv data source.
   */
  def from_csv(e: Column, schema: StructType, options: Map[String, String]): Column = withExpr {
    CsvToStruct(schema, options, e.expr)
  }

  /**
   * Parses a column containing a CSV string into a [[StructType]] with the specified schema.
   * Returns `null`, in the case of an unparseable string.
   * @group misc
   *
   * @param e a string column containing CSV data.
   * @param schema the schema to use when parsing the json string
   */
  def from_csv(e: Column, schema: StructType): Column =
    from_csv(e, schema, Map.empty[String, String])

  /**
   * Converts a column containing a [[StructType]] into a CSV string with the specified schema.
   * Throws an exception, in the case of an unsupported type.
   * @group misc
   *
   * @param e a struct column.
   * @param options options to control how the struct column is converted into a json string.
   *                accepts the same options and the json data source.
   */
  def to_csv(e: Column, options: Map[String, String]): Column = withExpr {
    StructToCsv(options, e.expr)
  }

  /**
   * Converts a column containing a [[StructType]] into a CSV string with the specified schema.
   * Throws an exception, in the case of an unsupported type.
   * @group misc
   *
   * @param e a struct column.
   */
  def to_csv(e: Column): Column = to_csv(e, Map.empty[String, String])

  /**
   * A convenient function to wrap an expression and produce a Column.
   */
  @inline private def withExpr(expr: Expression): Column = Column(expr)
}
