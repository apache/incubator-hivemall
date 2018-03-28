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

package org.apache.spark.sql.catalyst.plans.logical

import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.{Inner, JoinType}
import org.apache.spark.sql.types.{BooleanType, IntegerType}

case class JoinTopK(
    k: Int,
    left: LogicalPlan,
    right: LogicalPlan,
    joinType: JoinType,
    condition: Option[Expression])(
    val scoreExpr: NamedExpression,
    private[sql] val rankAttr: Seq[Attribute] = AttributeReference("rank", IntegerType)() :: Nil)
  extends BinaryNode with PredicateHelper {

  override def output: Seq[Attribute] = joinType match {
    case Inner => rankAttr ++ Seq(scoreExpr.toAttribute) ++ left.output ++ right.output
  }

  override def references: AttributeSet = {
    AttributeSet((expressions ++ Seq(scoreExpr)).flatMap(_.references))
  }

  override protected def validConstraints: Set[Expression] = joinType match {
    case Inner if condition.isDefined =>
      left.constraints.union(right.constraints)
        .union(splitConjunctivePredicates(condition.get).toSet)
  }

  override protected final def otherCopyArgs: Seq[AnyRef] = {
    scoreExpr :: rankAttr :: Nil
  }

  def duplicateResolved: Boolean = left.outputSet.intersect(right.outputSet).isEmpty

  lazy val resolvedExceptNatural: Boolean = {
    childrenResolved &&
      expressions.forall(_.resolved) &&
      duplicateResolved &&
      condition.forall(_.dataType == BooleanType)
  }

  override lazy val resolved: Boolean = joinType match {
    case Inner => resolvedExceptNatural
    case tpe => throw new AnalysisException(s"Unsupported using join type $tpe")
  }
}
