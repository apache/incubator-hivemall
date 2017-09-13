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
import org.apache.spark.sql.catalyst.utils.InternalRowPriorityQueue
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.metric._
import org.apache.spark.sql.types._

abstract class PriorityQueueShim {

  def insert(score: Any, row: InternalRow): Unit
  def get(): Iterator[InternalRow]
  def clear(): Unit
}

case class ShuffledHashJoinTopKExec(
    k: Int,
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    condition: Option[Expression],
    left: SparkPlan,
    right: SparkPlan)(
    scoreExpr: NamedExpression,
    rankAttr: Seq[Attribute])
  extends BinaryExecNode with TopKHelper with HashJoin with CodegenSupport {

  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"))

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

  private lazy val _priorityQueue = new PriorityQueueShim {

    private val q: InternalRowPriorityQueue = queue
    private val joinedRow = new JoinedRow

    override def insert(score: Any, row: InternalRow): Unit = {
      q += Tuple2(score, row)
    }

    override def get(): Iterator[InternalRow] = {
      val outputRows = queue.iterator.toSeq.reverse
      val (headScore, _) = outputRows.head
      val rankNum = outputRows.scanLeft((1, headScore)) { case ((rank, prevScore), (score, _)) =>
        if (prevScore == score) (rank, score) else (rank + 1, score)
      }
      val topKRow = new UnsafeRow(2)
      val bufferHolder = new BufferHolder(topKRow)
      val unsafeRowWriter = new UnsafeRowWriter(bufferHolder, 2)
      val scoreWriter = ScoreWriter(unsafeRowWriter, 1)
      outputRows.zip(rankNum.map(_._1)).map { case ((score, row), index) =>
        // Writes to an UnsafeRow directly
        bufferHolder.reset()
        unsafeRowWriter.write(0, index)
        scoreWriter.write(score)
        topKRow.setTotalSize(bufferHolder.totalSize())
        joinedRow.apply(topKRow, row)
      }.iterator
    }

    override def clear(): Unit = q.clear()
  }

  override def output: Seq[Attribute] = joinType match {
    case Inner => topKAttr ++ left.output ++ right.output
  }

  override protected final def otherCopyArgs: Seq[AnyRef] = {
    scoreExpr :: rankAttr :: Nil
  }

  override def requiredChildDistribution: Seq[Distribution] =
    ClusteredDistribution(leftKeys) :: ClusteredDistribution(rightKeys) :: Nil

  def buildHashedRelation(iter: Iterator[InternalRow]): HashedRelation = {
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
          _priorityQueue.insert(scoreProjection(resultRow).get(0, scoreType), resultRow)
        }
        val iter = _priorityQueue.get()
        _priorityQueue.clear()
        iter
      } else {
        Seq.empty
      }
    }
    val resultProj = createResultProjection()
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

  override def inputRDDs(): Seq[RDD[InternalRow]] = {
    left.execute() :: right.execute() :: Nil
  }

  // Accessor for generated code
  def priorityQueue(): PriorityQueueShim = _priorityQueue

  /**
   * Add a state of HashedRelation and return the variable name for it.
   */
  private def prepareHashedRelation(ctx: CodegenContext): String = {
    // create a name for HashedRelation
    val joinExec = ctx.addReferenceObj("joinExec", this)
    val relationTerm = ctx.freshName("relation")
    val clsName = HashedRelation.getClass.getName.replace("$", "")
    ctx.addMutableState(clsName, relationTerm,
      s"""
         | $relationTerm = ($clsName) $joinExec.buildHashedRelation(inputs[1]);
         | incPeakExecutionMemory($relationTerm.estimatedSize());
       """.stripMargin)
    relationTerm
  }

  /**
   * Creates variables for left part of result row.
   *
   * In order to defer the access after condition and also only access once in the loop,
   * the variables should be declared separately from accessing the columns, we can't use the
   * codegen of BoundReference here.
   */
  private def createLeftVars(ctx: CodegenContext, leftRow: String): Seq[ExprCode] = {
    ctx.INPUT_ROW = leftRow
    left.output.zipWithIndex.map { case (a, i) =>
      val value = ctx.freshName("value")
      val valueCode = ctx.getValue(leftRow, a.dataType, i.toString)
      // declare it as class member, so we can access the column before or in the loop.
      ctx.addMutableState(ctx.javaType(a.dataType), value, "")
      if (a.nullable) {
        val isNull = ctx.freshName("isNull")
        ctx.addMutableState("boolean", isNull, "")
        val code =
          s"""
             |$isNull = $leftRow.isNullAt($i);
             |$value = $isNull ? ${ctx.defaultValue(a.dataType)} : ($valueCode);
           """.stripMargin
        ExprCode(code, isNull, value)
      } else {
        ExprCode(s"$value = $valueCode;", "false", value)
      }
    }
  }

  /**
   * Creates the variables for right part of result row, using BoundReference, since the right
   * part are accessed inside the loop.
   */
  private def createRightVar(ctx: CodegenContext, rightRow: String): Seq[ExprCode] = {
    ctx.INPUT_ROW = rightRow
    right.output.zipWithIndex.map { case (a, i) =>
      BoundReference(i, a.dataType, a.nullable).genCode(ctx)
    }
  }

  /**
   * Returns the code for generating join key for stream side, and expression of whether the key
   * has any null in it or not.
   */
  private def genStreamSideJoinKey(ctx: CodegenContext, leftRow: String): (ExprCode, String) = {
    ctx.INPUT_ROW = leftRow
    if (streamedKeys.length == 1 && streamedKeys.head.dataType == LongType) {
      // generate the join key as Long
      val ev = streamedKeys.head.genCode(ctx)
      (ev, ev.isNull)
    } else {
      // generate the join key as UnsafeRow
      val ev = GenerateUnsafeProjection.createCode(ctx, streamedKeys)
      (ev, s"${ev.value}.anyNull()")
    }
  }

  private def createScoreVar(ctx: CodegenContext, row: String): ExprCode = {
    ctx.INPUT_ROW = row
    BindReferences.bindReference(scoreExpr, left.output ++ right.output).genCode(ctx)
  }

  private def createResultVars(ctx: CodegenContext, resultRow: String): Seq[ExprCode] = {
    ctx.INPUT_ROW = resultRow
    output.zipWithIndex.map { case (a, i) =>
      val value = ctx.freshName("value")
      val valueCode = ctx.getValue(resultRow, a.dataType, i.toString)
      // declare it as class member, so we can access the column before or in the loop.
      ctx.addMutableState(ctx.javaType(a.dataType), value, "")
      if (a.nullable) {
        val isNull = ctx.freshName("isNull")
        ctx.addMutableState("boolean", isNull, "")
        val code =
          s"""
             |$isNull = $resultRow.isNullAt($i);
             |$value = $isNull ? ${ctx.defaultValue(a.dataType)} : ($valueCode);
           """.stripMargin
        ExprCode(code, isNull, value)
      } else {
        ExprCode(s"$value = $valueCode;", "false", value)
      }
    }
  }

  /**
   * Splits variables based on whether it's used by condition or not, returns the code to create
   * these variables before the condition and after the condition.
   *
   * Only a few columns are used by condition, then we can skip the accessing of those columns
   * that are not used by condition also filtered out by condition.
   */
  private def splitVarsByCondition(
      attributes: Seq[Attribute],
      variables: Seq[ExprCode]): (String, String) = {
    if (condition.isDefined) {
      val condRefs = condition.get.references
      val (used, notUsed) = attributes.zip(variables).partition{ case (a, ev) =>
        condRefs.contains(a)
      }
      val beforeCond = evaluateVariables(used.map(_._2))
      val afterCond = evaluateVariables(notUsed.map(_._2))
      (beforeCond, afterCond)
    } else {
      (evaluateVariables(variables), "")
    }
  }

  override def doProduce(ctx: CodegenContext): String = {
    ctx.copyResult = true

    val topKJoin = ctx.addReferenceObj("topKJoin", this)

    // Prepare a priority queue for top-K computing
    val pQueue = ctx.freshName("queue")
    ctx.addMutableState(classOf[PriorityQueueShim].getName, pQueue,
      s"$pQueue = $topKJoin.priorityQueue();")

    // Prepare variables for a left side
    val leftIter = ctx.freshName("leftIter")
    ctx.addMutableState("scala.collection.Iterator", leftIter, s"$leftIter = inputs[0];")
    val leftRow = ctx.freshName("leftRow")
    ctx.addMutableState("InternalRow", leftRow, "")
    val leftVars = createLeftVars(ctx, leftRow)

    // Prepare variables for a right side
    val rightRow = ctx.freshName("rightRow")
    val rightVars = createRightVar(ctx, rightRow)

    // Build a hashed relation from a right side
    val buildRelation = prepareHashedRelation(ctx)

    // Project join keys from a left side
    val (keyEv, anyNull) = genStreamSideJoinKey(ctx, leftRow)

    // Prepare variables for joined rows
    val joinedRow = ctx.freshName("joinedRow")
    val joinedRowCls = classOf[JoinedRow].getName
    ctx.addMutableState(joinedRowCls, joinedRow, s"$joinedRow = new $joinedRowCls();")

    // Project score values from joined rows
    val scoreVar = createScoreVar(ctx, joinedRow)

    // Prepare variables for output rows
    val resultRow = ctx.freshName("resultRow")
    val resultVars = createResultVars(ctx, resultRow)

    val (beforeLoop, condCheck) = if (condition.isDefined) {
      // Split the code of creating variables based on whether it's used by condition or not.
      val loaded = ctx.freshName("loaded")
      val (leftBefore, leftAfter) = splitVarsByCondition(left.output, leftVars)
      val (rightBefore, rightAfter) = splitVarsByCondition(right.output, rightVars)
      // Generate code for condition
      ctx.currentVars = leftVars ++ rightVars
      val cond = BindReferences.bindReference(condition.get, output).genCode(ctx)
      // evaluate the columns those used by condition before loop
      val before = s"""
           |boolean $loaded = false;
           |$leftBefore
         """.stripMargin

      val checking = s"""
         |$rightBefore
         |${cond.code}
         |if (${cond.isNull} || !${cond.value}) continue;
         |if (!$loaded) {
         |  $loaded = true;
         |  $leftAfter
         |}
         |$rightAfter
     """.stripMargin
      (before, checking)
    } else {
      (evaluateVariables(leftVars), "")
    }

    val numOutput = metricTerm(ctx, "numOutputRows")

    val matches = ctx.freshName("matches")
    val topKRows = ctx.freshName("topKRows")
    val iteratorCls = classOf[Iterator[UnsafeRow]].getName

    s"""
       |$leftRow = null;
       |while ($leftIter.hasNext()) {
       |  $leftRow = (InternalRow) $leftIter.next();
       |
       |  // Generate join key for stream side
       |  ${keyEv.code}
       |
       |  // Find matches from HashedRelation
       |  $iteratorCls $matches = $anyNull? null : ($iteratorCls)$buildRelation.get(${keyEv.value});
       |  if ($matches == null) continue;
       |
       |  // Join top-K right rows
       |  while ($matches.hasNext()) {
       |    ${beforeLoop.trim}
       |    InternalRow $rightRow = (InternalRow) $matches.next();
       |    ${condCheck.trim}
       |    InternalRow row = $joinedRow.apply($leftRow, $rightRow);
       |    // Compute a score for the `row`
       |    ${scoreVar.code}
       |    $pQueue.insert(${scoreVar.value}, row);
       |  }
       |
       |  // Get top-K rows
       |  $iteratorCls $topKRows = $pQueue.get();
       |  $pQueue.clear();
       |
       |  // Output top-K rows
       |  while ($topKRows.hasNext()) {
       |    InternalRow $resultRow = (InternalRow) $topKRows.next();
       |    $numOutput.add(1);
       |    ${consume(ctx, resultVars)}
       |  }
       |
       |  if (shouldStop()) return;
       |}
     """.stripMargin
  }
}
