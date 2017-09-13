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

import org.apache.spark.ml.linalg.{BLAS, DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object HivemallUtils {

  // # of maximum dimensions for feature vectors
  private[this] val maxDims = 100000000

  /**
   * Check whether the given schema contains a column of the required data type.
   * @param colName  column name
   * @param dataType  required column data type
   */
  private[this] def checkColumnType(schema: StructType, colName: String, dataType: DataType)
    : Unit = {
    val actualDataType = schema(colName).dataType
    require(actualDataType.equals(dataType),
      s"Column $colName must be of type $dataType but was actually $actualDataType.")
  }

  def to_vector_func(dense: Boolean, dims: Int): Seq[String] => Vector = {
    if (dense) {
      // Dense features
      i: Seq[String] => {
        val features = new Array[Double](dims)
        i.map { ft =>
          val s = ft.split(":").ensuring(_.size == 2)
          features(s(0).toInt) = s(1).toDouble
        }
        Vectors.dense(features)
      }
    } else {
      // Sparse features
      i: Seq[String] => {
        val features = i.map { ft =>
          // val s = ft.split(":").ensuring(_.size == 2)
          val s = ft.split(":")
          (s(0).toInt, s(1).toDouble)
        }
        Vectors.sparse(dims, features)
      }
    }
  }

  def to_hivemall_features_func(): Vector => Array[String] = {
    case dv: DenseVector =>
      dv.values.zipWithIndex.map {
        case (value, index) => s"$index:$value"
      }
    case sv: SparseVector =>
      sv.values.zip(sv.indices).map {
        case (value, index) => s"$index:$value"
      }
    case v =>
      throw new IllegalArgumentException(s"Do not support vector type ${v.getClass}")
  }

  def append_bias_func(): Vector => Vector = {
    case dv: DenseVector =>
      val inputValues = dv.values
      val inputLength = inputValues.length
      val outputValues = Array.ofDim[Double](inputLength + 1)
      System.arraycopy(inputValues, 0, outputValues, 0, inputLength)
      outputValues(inputLength) = 1.0
      Vectors.dense(outputValues)
    case sv: SparseVector =>
      val inputValues = sv.values
      val inputIndices = sv.indices
      val inputValuesLength = inputValues.length
      val dim = sv.size
      val outputValues = Array.ofDim[Double](inputValuesLength + 1)
      val outputIndices = Array.ofDim[Int](inputValuesLength + 1)
      System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
      System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)
      outputValues(inputValuesLength) = 1.0
      outputIndices(inputValuesLength) = dim
      Vectors.sparse(dim + 1, outputIndices, outputValues)
    case v =>
      throw new IllegalArgumentException(s"Do not support vector type ${v.getClass}")
  }

  /**
   * Transforms Hivemall features into a [[Vector]].
   */
  def to_vector(dense: Boolean = false, dims: Int = maxDims): UserDefinedFunction = {
    udf(to_vector_func(dense, dims))
  }

  /**
   * Transforms a [[Vector]] into Hivemall features.
   */
  def to_hivemall_features: UserDefinedFunction = udf(to_hivemall_features_func)

  /**
   * Returns a new [[Vector]] with `1.0` (bias) appended to the input [[Vector]].
   * @group ftvec
   */
  def append_bias: UserDefinedFunction = udf(append_bias_func)

  /**
   * Builds a [[Vector]]-based model from a table of Hivemall models
   */
  def vectorized_model(df: DataFrame, dense: Boolean = false, dims: Int = maxDims)
    : UserDefinedFunction = {
    checkColumnType(df.schema, "feature", StringType)
    checkColumnType(df.schema, "weight", DoubleType)

    import df.sqlContext.implicits._
    val intercept = df
      .where($"feature" === "0")
      .select($"weight")
      .map { case Row(weight: Double) => weight}
      .reduce(_ + _)
    val weights = to_vector_func(dense, dims)(
      df.select($"feature", $"weight")
        .where($"feature" !== "0")
        .map { case Row(label: String, feature: Double) => s"${label}:$feature"}
        .collect.toSeq)

    udf((input: Vector) => BLAS.dot(input, weights) + intercept)
  }
}
