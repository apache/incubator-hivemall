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

import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.RelationalGroupedDataset
import org.apache.spark.sql.catalyst.analysis.UnresolvedAlias
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.logical.Aggregate
import org.apache.spark.sql.catalyst.plans.logical.Pivot
import org.apache.spark.sql.hive.HiveShim.HiveFunctionWrapper
import org.apache.spark.sql.types._

/**
 * Groups the [[DataFrame]] using the specified columns, so we can run aggregation on them.
 *
 * @groupname classifier
 * @groupname ensemble
 * @groupname evaluation
 * @groupname topicmodel
 * @groupname ftvec.selection
 * @groupname ftvec.text
 * @groupname ftvec.trans
 * @groupname tools.array
 * @groupname tools.bits
 * @groupname tools.list
 * @groupname tools.map
 * @groupname tools.matrix
 * @groupname tools.math
 *
 * A list of unsupported functions is as follows:
 *  * ftvec.conv
 *   - conv2dense
 *   - build_bins
 */
final class HivemallGroupedDataset(groupBy: RelationalGroupedDataset) {

  /**
   * @see hivemall.classifier.KPAPredictUDAF
   * @group classifier
   */
  def kpa_predict(xh: String, xk: String, w0: String, w1: String, w2: String, w3: String)
    : DataFrame = {
    checkType(xh, DoubleType)
    checkType(xk, DoubleType)
    checkType(w0, FloatType)
    checkType(w1, FloatType)
    checkType(w2, FloatType)
    checkType(w3, FloatType)
    val udaf = HiveUDAFFunction(
        "kpa_predict",
        new HiveFunctionWrapper("hivemall.classifier.KPAPredictUDAF"),
        Seq(xh, xk, w0, w1, w2, w3).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ensemble.bagging.VotedAvgUDAF
   * @group ensemble
   */
  def voted_avg(weight: String): DataFrame = {
    checkType(weight, DoubleType)
    val udaf = HiveUDAFFunction(
        "voted_avg",
        new HiveFunctionWrapper("hivemall.ensemble.bagging.WeightVotedAvgUDAF"),
        Seq(weight).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ensemble.bagging.WeightVotedAvgUDAF
   * @group ensemble
   */
  def weight_voted_avg(weight: String): DataFrame = {
    checkType(weight, DoubleType)
    val udaf = HiveUDAFFunction(
        "weight_voted_avg",
        new HiveFunctionWrapper("hivemall.ensemble.bagging.WeightVotedAvgUDAF"),
        Seq(weight).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ensemble.ArgminKLDistanceUDAF
   * @group ensemble
   */
  def argmin_kld(weight: String, conv: String): DataFrame = {
    checkType(weight, FloatType)
    checkType(conv, FloatType)
    val udaf = HiveUDAFFunction(
        "argmin_kld",
        new HiveFunctionWrapper("hivemall.ensemble.ArgminKLDistanceUDAF"),
        Seq(weight, conv).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ensemble.MaxValueLabelUDAF"
   * @group ensemble
   */
  def max_label(score: String, label: String): DataFrame = {
    // checkType(score, DoubleType)
    checkType(label, StringType)
    val udaf = HiveUDAFFunction(
        "max_label",
        new HiveFunctionWrapper("hivemall.ensemble.MaxValueLabelUDAF"),
        Seq(score, label).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ensemble.MaxRowUDAF
   * @group ensemble
   */
  def maxrow(score: String, label: String): DataFrame = {
    checkType(score, DoubleType)
    checkType(label, StringType)
    val udaf = HiveUDAFFunction(
        "maxrow",
        new HiveFunctionWrapper("hivemall.ensemble.MaxRowUDAF"),
        Seq(score, label).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.smile.tools.RandomForestEnsembleUDAF
   * @group ensemble
   */
  @scala.annotation.varargs
  def rf_ensemble(yhat: String, others: String*): DataFrame = {
    checkType(yhat, IntegerType)
    val udaf = HiveUDAFFunction(
        "rf_ensemble",
        new HiveFunctionWrapper("hivemall.smile.tools.RandomForestEnsembleUDAF"),
        (yhat +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.MeanAbsoluteErrorUDAF
   * @group evaluation
   */
  def mae(predict: String, target: String): DataFrame = {
    checkType(predict, DoubleType)
    checkType(target, DoubleType)
    val udaf = HiveUDAFFunction(
        "mae",
        new HiveFunctionWrapper("hivemall.evaluation.MeanAbsoluteErrorUDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.MeanSquareErrorUDAF
   * @group evaluation
   */
  def mse(predict: String, target: String): DataFrame = {
    checkType(predict, DoubleType)
    checkType(target, DoubleType)
    val udaf = HiveUDAFFunction(
        "mse",
        new HiveFunctionWrapper("hivemall.evaluation.MeanSquaredErrorUDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.RootMeanSquareErrorUDAF
   * @group evaluation
   */
  def rmse(predict: String, target: String): DataFrame = {
    checkType(predict, DoubleType)
    checkType(target, DoubleType)
    val udaf = HiveUDAFFunction(
        "rmse",
        new HiveFunctionWrapper("hivemall.evaluation.RootMeanSquaredErrorUDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.R2UDAF
   * @group evaluation
   */
  def r2(predict: String, target: String): DataFrame = {
    checkType(predict, DoubleType)
    checkType(target, DoubleType)
    val udaf = HiveUDAFFunction(
        "r2",
        new HiveFunctionWrapper("hivemall.evaluation.R2UDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.LogarithmicLossUDAF
   * @group evaluation
   */
  def logloss(predict: String, target: String): DataFrame = {
    checkType(predict, DoubleType)
    checkType(target, DoubleType)
    val udaf = HiveUDAFFunction(
        "logloss",
        new HiveFunctionWrapper("hivemall.evaluation.LogarithmicLossUDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.F1ScoreUDAF
   * @group evaluation
   */
  def f1score(predict: String, target: String): DataFrame = {
    // checkType(target, ArrayType(IntegerType, false))
    // checkType(predict, ArrayType(IntegerType, false))
    val udaf = HiveUDAFFunction(
        "f1score",
        new HiveFunctionWrapper("hivemall.evaluation.F1ScoreUDAF"),
        Seq(predict, target).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.NDCGUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def ndcg(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "ndcg",
        new HiveFunctionWrapper("hivemall.evaluation.NDCGUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.PrecisionUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def precision_at(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "precision_at",
        new HiveFunctionWrapper("hivemall.evaluation.PrecisionUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.RecallUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def recall_at(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "recall_at",
        new HiveFunctionWrapper("hivemall.evaluation.RecallUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.HitRateUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def hitrate(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "hitrate",
        new HiveFunctionWrapper("hivemall.evaluation.HitRateUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.MRRUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def mrr(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "mrr",
        new HiveFunctionWrapper("hivemall.evaluation.MRRUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.MAPUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def average_precision(rankItems: String, correctItems: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "average_precision",
        new HiveFunctionWrapper("hivemall.evaluation.MAPUDAF"),
        (rankItems +: correctItems +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.evaluation.AUCUDAF
   * @group evaluation
   */
  @scala.annotation.varargs
  def auc(args: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "auc",
        new HiveFunctionWrapper("hivemall.evaluation.AUCUDAF"),
        args.map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.topicmodel.LDAPredictUDAF
   * @group topicmodel
   */
  @scala.annotation.varargs
  def lda_predict(word: String, value: String, label: String, lambda: String, others: String*)
    : DataFrame = {
    checkType(word, StringType)
    checkType(value, DoubleType)
    checkType(label, IntegerType)
    checkType(lambda, DoubleType)
    val udaf = HiveUDAFFunction(
        "lda_predict",
        new HiveFunctionWrapper("hivemall.topicmodel.LDAPredictUDAF"),
        (word +: value +: label +: lambda +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.topicmodel.PLSAPredictUDAF
   * @group topicmodel
   */
  @scala.annotation.varargs
  def plsa_predict(word: String, value: String, label: String, prob: String, others: String*)
    : DataFrame = {
    checkType(word, StringType)
    checkType(value, DoubleType)
    checkType(label, IntegerType)
    checkType(prob, DoubleType)
    val udaf = HiveUDAFFunction(
        "plsa_predict",
        new HiveFunctionWrapper("hivemall.topicmodel.PLSAPredictUDAF"),
        (word +: value +: label +: prob +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ftvec.text.TermFrequencyUDAF
   * @group ftvec.text
   */
  def tf(text: String): DataFrame = {
    checkType(text, StringType)
    val udaf = HiveUDAFFunction(
        "tf",
        new HiveFunctionWrapper("hivemall.ftvec.text.TermFrequencyUDAF"),
        Seq(text).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ftvec.trans.OnehotEncodingUDAF
   * @group ftvec.trans
   */
  @scala.annotation.varargs
  def onehot_encoding(feature: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "onehot_encoding",
        new HiveFunctionWrapper("hivemall.ftvec.trans.OnehotEncodingUDAF"),
        (feature +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.ftvec.selection.SignalNoiseRatioUDAF
   * @group ftvec.selection
   */
  def snr(feature: String, label: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "snr",
        new HiveFunctionWrapper("hivemall.ftvec.selection.SignalNoiseRatioUDAF"),
        Seq(feature, label).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.array.ArrayAvgGenericUDAF
   * @group tools.array
   */
  def array_avg(ar: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "array_avg",
        new HiveFunctionWrapper("hivemall.tools.array.ArrayAvgGenericUDAF"),
        Seq(ar).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.array.ArraySumUDAF
   * @group tools.array
   */
  def array_sum(ar: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "array_sum",
        new HiveFunctionWrapper("hivemall.tools.array.ArraySumUDAF"),
        Seq(ar).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.bits.BitsCollectUDAF
   * @group tools.bits
   */
  def bits_collect(x: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "bits_collect",
        new HiveFunctionWrapper("hivemall.tools.bits.BitsCollectUDAF"),
        Seq(x).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.list.UDAFToOrderedList
   * @group tools.list
   */
  @scala.annotation.varargs
  def to_ordered_list(value: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "to_ordered_list",
        new HiveFunctionWrapper("hivemall.tools.list.UDAFToOrderedList"),
        (value +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.map.UDAFToMap
   * @group tools.map
   */
  def to_map(key: String, value: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "to_map",
        new HiveFunctionWrapper("hivemall.tools.map.UDAFToMap"),
        Seq(key, value).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.map.UDAFToOrderedMap
   * @group tools.map
   */
  @scala.annotation.varargs
  def to_ordered_map(key: String, value: String, others: String*): DataFrame = {
    val udaf = HiveUDAFFunction(
        "to_ordered_map",
        new HiveFunctionWrapper("hivemall.tools.map.UDAFToOrderedMap"),
        (key +: value +: others).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.matrix.TransposeAndDotUDAF
   * @group tools.matrix
   */
  def transpose_and_dot(matrix0_row: String, matrix1_row: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "transpose_and_dot",
        new HiveFunctionWrapper("hivemall.tools.matrix.TransposeAndDotUDAF"),
        Seq(matrix0_row, matrix1_row).map(df(_).expr),
        isUDAFBridgeRequired = false)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * @see hivemall.tools.math.L2NormUDAF
   * @group tools.math
   */
  def l2_norm(xi: String): DataFrame = {
    val udaf = HiveUDAFFunction(
        "l2_norm",
        new HiveFunctionWrapper("hivemall.tools.math.L2NormUDAF"),
        Seq(xi).map(df(_).expr),
        isUDAFBridgeRequired = true)
      .toAggregateExpression()
    toDF(Alias(udaf, udaf.prettyName)() :: Nil)
  }

  /**
   * [[RelationalGroupedDataset]] has the three values as private fields, so, to inject Hivemall
   * aggregate functions, we fetch them via Java Reflections.
   */
  private val df = getPrivateField[DataFrame]("org$apache$spark$sql$RelationalGroupedDataset$$df")
  private val groupingExprs = getPrivateField[Seq[Expression]]("groupingExprs")
  private val groupType = getPrivateField[RelationalGroupedDataset.GroupType]("groupType")

  private def getPrivateField[T](name: String): T = {
    val field = groupBy.getClass.getDeclaredField(name)
    field.setAccessible(true)
    field.get(groupBy).asInstanceOf[T]
  }

  private def toDF(aggExprs: Seq[Expression]): DataFrame = {
    val aggregates = if (df.sqlContext.conf.dataFrameRetainGroupColumns) {
      groupingExprs ++ aggExprs
    } else {
      aggExprs
    }

    val aliasedAgg = aggregates.map(alias)

    groupType match {
      case RelationalGroupedDataset.GroupByType =>
        Dataset.ofRows(
          df.sparkSession, Aggregate(groupingExprs, aliasedAgg, df.logicalPlan))
      case RelationalGroupedDataset.RollupType =>
        Dataset.ofRows(
          df.sparkSession, Aggregate(Seq(Rollup(groupingExprs)), aliasedAgg, df.logicalPlan))
      case RelationalGroupedDataset.CubeType =>
        Dataset.ofRows(
          df.sparkSession, Aggregate(Seq(Cube(groupingExprs)), aliasedAgg, df.logicalPlan))
      case RelationalGroupedDataset.PivotType(pivotCol, values) =>
        val aliasedGrps = groupingExprs.map(alias)
        Dataset.ofRows(
          df.sparkSession, Pivot(aliasedGrps, pivotCol, values, aggExprs, df.logicalPlan))
    }
  }

  private def alias(expr: Expression): NamedExpression = expr match {
    case u: UnresolvedAttribute => UnresolvedAlias(u)
    case expr: NamedExpression => expr
    case expr: Expression => Alias(expr, expr.prettyName)()
  }

  private def checkType(colName: String, expected: DataType) = {
    val dataType = df.resolve(colName).dataType
    if (dataType != expected) {
      throw new AnalysisException(
        s""""$colName" must be $expected, however it is $dataType""")
    }
  }
}

object HivemallGroupedDataset {

  /**
   * Implicitly inject the [[HivemallGroupedDataset]] into [[RelationalGroupedDataset]].
   */
  implicit def relationalGroupedDatasetToHivemallOne(
      groupBy: RelationalGroupedDataset): HivemallGroupedDataset = {
    new HivemallGroupedDataset(groupBy)
  }
}
