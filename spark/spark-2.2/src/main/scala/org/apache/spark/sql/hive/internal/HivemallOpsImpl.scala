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

package org.apache.spark.sql.hive.internal

import org.apache.spark.internal.Logging
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.plans.logical.{Generate, LogicalPlan}
import org.apache.spark.sql.hive._
import org.apache.spark.sql.hive.HiveShim.HiveFunctionWrapper

/**
 * This is an implementation class for [[org.apache.spark.sql.hive.HivemallOps]].
 * This class mainly uses the internal Spark classes (e.g., `Generate` and `HiveGenericUDTF`) that
 * have unstable interfaces (so, these interfaces may evolve in upcoming releases).
 * Therefore, the objective of this class is to extract these unstable parts
 * from [[org.apache.spark.sql.hive.HivemallOps]].
 */
private[hive] object HivemallOpsImpl extends Logging {

  def planHiveUDF(
      className: String,
      funcName: String,
      argumentExprs: Seq[Column]): Expression = {
    HiveSimpleUDF(
      name = funcName,
      funcWrapper = new HiveFunctionWrapper(className),
      children = argumentExprs.map(_.expr)
     )
  }

  def planHiveGenericUDF(
      className: String,
      funcName: String,
      argumentExprs: Seq[Column]): Expression = {
    HiveGenericUDF(
      name = funcName,
      funcWrapper = new HiveFunctionWrapper(className),
      children = argumentExprs.map(_.expr)
     )
  }

  def planHiveGenericUDTF(
      df: DataFrame,
      className: String,
      funcName: String,
      argumentExprs: Seq[Column],
      outputAttrNames: Seq[String]): LogicalPlan = {
    Generate(
      generator = HiveGenericUDTF(
        name = funcName,
        funcWrapper = new HiveFunctionWrapper(className),
        children = argumentExprs.map(_.expr)
      ),
      join = false,
      outer = false,
      qualifier = None,
      generatorOutput = outputAttrNames.map(UnresolvedAttribute(_)),
      child = df.logicalPlan)
  }
}
