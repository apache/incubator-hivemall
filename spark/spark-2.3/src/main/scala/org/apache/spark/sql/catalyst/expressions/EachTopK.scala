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

package org.apache.spark.sql.catalyst.expressions

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.util.TypeUtils
import org.apache.spark.sql.catalyst.utils.InternalRowPriorityQueue
import org.apache.spark.sql.types._

trait TopKHelper {

  def k: Int
  def scoreType: DataType

  @transient val ScoreTypes = TypeCollection(
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType
  )

  protected case class ScoreWriter(writer: UnsafeRowWriter, ordinal: Int) {

    def write(v: Any): Unit = scoreType match {
      case ByteType => writer.write(ordinal, v.asInstanceOf[Byte])
      case ShortType => writer.write(ordinal, v.asInstanceOf[Short])
      case IntegerType => writer.write(ordinal, v.asInstanceOf[Int])
      case LongType => writer.write(ordinal, v.asInstanceOf[Long])
      case FloatType => writer.write(ordinal, v.asInstanceOf[Float])
      case DoubleType => writer.write(ordinal, v.asInstanceOf[Double])
      case d: DecimalType => writer.write(ordinal, v.asInstanceOf[Decimal], d.precision, d.scale)
    }
  }

  protected lazy val scoreOrdering = {
    val ordering = TypeUtils.getInterpretedOrdering(scoreType)
    if (k > 0) ordering else ordering.reverse
  }

  protected lazy val reverseScoreOrdering = scoreOrdering.reverse

  protected lazy val queue: InternalRowPriorityQueue = {
    new InternalRowPriorityQueue(Math.abs(k), (x: Any, y: Any) => scoreOrdering.compare(x, y))
  }
}

case class EachTopK(
    k: Int,
    scoreExpr: Expression,
    groupExprs: Seq[Expression],
    elementSchema: StructType,
    children: Seq[Attribute])
  extends Generator with TopKHelper with CodegenFallback {

  override val scoreType: DataType = scoreExpr.dataType

  private lazy val groupingProjection: UnsafeProjection = UnsafeProjection.create(groupExprs)
  private lazy val scoreProjection: UnsafeProjection = UnsafeProjection.create(scoreExpr :: Nil)

  // The grouping key of the current partition
  private var currentGroupingKeys: UnsafeRow = _

  override def checkInputDataTypes(): TypeCheckResult = {
    if (!ScoreTypes.acceptsType(scoreExpr.dataType)) {
      TypeCheckResult.TypeCheckFailure(s"$scoreExpr must have a comparable type")
    } else {
      TypeCheckResult.TypeCheckSuccess
    }
  }

  private def topKRowsForGroup(): Seq[InternalRow] = if (queue.size > 0) {
    val outputRows = queue.iterator.toSeq.reverse
    val (headScore, _) = outputRows.head
    val rankNum = outputRows.scanLeft((1, headScore)) { case ((rank, prevScore), (score, _)) =>
      if (prevScore == score) (rank, score) else (rank + 1, score)
    }
    val topKRow = new UnsafeRow(1)
    val bufferHolder = new BufferHolder(topKRow)
    val unsafeRowWriter = new UnsafeRowWriter(bufferHolder, 1)
    outputRows.zip(rankNum.map(_._1)).map { case ((_, row), index) =>
      // Writes to an UnsafeRow directly
      bufferHolder.reset()
      unsafeRowWriter.write(0, index)
      topKRow.setTotalSize(bufferHolder.totalSize())
      new JoinedRow(topKRow, row)
    }
  } else {
    Seq.empty
  }

  override def eval(input: InternalRow): TraversableOnce[InternalRow] = {
    val groupingKeys = groupingProjection(input)
    val ret = if (currentGroupingKeys != groupingKeys) {
      val topKRows = topKRowsForGroup()
      currentGroupingKeys = groupingKeys.copy()
      queue.clear()
      topKRows
    } else {
      Iterator.empty
    }
    queue += Tuple2(scoreProjection(input).get(0, scoreType), input)
    ret
  }

  override def terminate(): TraversableOnce[InternalRow] = {
    if (queue.size > 0) {
      val topKRows = topKRowsForGroup()
      queue.clear()
      topKRows
    } else {
      Iterator.empty
    }
  }

  // TODO: Need to support codegen
  // protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode
}
