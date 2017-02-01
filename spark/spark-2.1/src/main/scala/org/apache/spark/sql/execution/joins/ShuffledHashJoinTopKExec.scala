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
package org.apache.spark.sql.execution.joins

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.metric._
import org.apache.spark.sql.types._

// TODO: Need to support codegen
case class ShuffledHashJoinTopKExec(
    k: Int,
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)(
    scoreExpr: NamedExpression,
    rankAttr: Seq[Attribute])
  extends BinaryExecNode with TopKHelper with HashJoin {

  override val scoreType: DataType = scoreExpr.dataType
  override val joinType: JoinType = Inner
  override val buildSide: BuildSide = BuildRight // Only support `BuildRight`

  private lazy val scoreProjection: UnsafeProjection =
    UnsafeProjection.create(scoreExpr :: Nil, left.output ++ right.output)

  private lazy val boundCondition = if (condition.isDefined) {
    (r: InternalRow) => newPredicate(condition.get, streamedPlan.output ++ buildPlan.output).eval(r)
  } else {
    (r: InternalRow) => true
  }

  private lazy val topKAttr = rankAttr :+ scoreExpr.toAttribute

  override def output: Seq[Attribute] = joinType match {
    case Inner => topKAttr ++ left.output ++ right.output
  }

  override protected final def otherCopyArgs: Seq[AnyRef] = {
    scoreExpr :: rankAttr :: Nil
  }

  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(leftKeys) :: ClusteredDistribution(rightKeys) :: Nil

  private def buildHashedRelation(iter: Iterator[InternalRow]): HashedRelation = {
    val context = TaskContext.get()
    val relation = HashedRelation(iter, buildKeys, taskMemoryManager = context.taskMemoryManager())
    context.addTaskCompletionListener(_ => relation.close())
    relation
  }

  override protected def createResultProjection(): (InternalRow) => InternalRow = joinType match {
    case Inner =>
      // Always put the stream side on left to simplify implementation
      // both of left and right side could be null
      UnsafeProjection.create(
        output, (topKAttr ++ streamedPlan.output ++ buildPlan.output).map(_.withNullability(true)))
  }

  protected def InnerJoin(
      streamedIter: Iterator[InternalRow],
      hashedRelation: HashedRelation,
      numOutputRows: SQLMetric): Iterator[InternalRow] = {
    val joinRow = new JoinedRow
    val joinKeysProj = streamSideKeyGenerator()
    val joinedIter = streamedIter.flatMap { srow =>
      joinRow.withLeft(srow)
      val joinKeys = joinKeysProj(srow) // `joinKeys` is also a grouping key
      val matches = hashedRelation.get(joinKeys)
      if (matches != null) {
        matches.map(joinRow.withRight).filter(boundCondition).foreach { resultRow =>
          queue += Tuple2(scoreProjection(resultRow).get(0, scoreType), resultRow)
        }
        val topKRow = new UnsafeRow(2)
        val bufferHolder = new BufferHolder(topKRow)
        val unsafeRowWriter = new UnsafeRowWriter(bufferHolder, 2)
        val scoreWriter = ScoreWriter(unsafeRowWriter, 1)
        val iter = queue.iterator.toSeq.sortBy(_._1)(reverseScoreOrdering)
          .zipWithIndex.map { case ((score, row), index) =>
            // Writes to an UnsafeRow directly
            bufferHolder.reset()
            unsafeRowWriter.write(0, 1 + index)
            scoreWriter.write(score)
            topKRow.setTotalSize(bufferHolder.totalSize())
            new JoinedRow(topKRow, row)
          }
        queue.clear
        iter
      } else {
        Seq.empty
      }
    }
    val resultProj = createResultProjection
    (joinedIter ++ queue.iterator.toSeq.sortBy(_._1)(reverseScoreOrdering)
      .map(_._2)).map { r =>
      resultProj(r)
    }
  }

  override protected def doExecute(): RDD[InternalRow] = {
    streamedPlan.execute().zipPartitions(buildPlan.execute()) { (streamIter, buildIter) =>
      val hashed = buildHashedRelation(buildIter)
      InnerJoin(streamIter, hashed, null)
    }
  }
}
