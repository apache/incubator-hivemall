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

package org.apache.spark.sql.hive.benchmark

import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.execution.benchmark.BenchmarkBase
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HivemallOps._
import org.apache.spark.sql.hive.internal.HivemallOpsImpl._
import org.apache.spark.sql.types._
import org.apache.spark.test.TestUtils
import org.apache.spark.util.Benchmark

class TestFuncWrapper(df: DataFrame) {

  def hive_each_top_k(k: Column, group: Column, value: Column, args: Column*)
    : DataFrame = withTypedPlan {
    planHiveGenericUDTF(
      df.repartition(group).sortWithinPartitions(group),
      "hivemall.tools.EachTopKUDTF",
      "each_top_k",
      Seq(k, group, value) ++ args,
      Seq("rank", "key") ++ args.map { _.expr match {
        case ua: UnresolvedAttribute => ua.name
      }}
    )
  }

  /**
   * A convenient function to wrap a logical plan and produce a DataFrame.
   */
  @inline private[this] def withTypedPlan(logicalPlan: => LogicalPlan): DataFrame = {
    val queryExecution = df.sparkSession.sessionState.executePlan(logicalPlan)
    val outputSchema = queryExecution.sparkPlan.schema
    new Dataset[Row](df.sparkSession, queryExecution, RowEncoder(outputSchema))
  }
}

object TestFuncWrapper {

  /**
   * Implicitly inject the [[TestFuncWrapper]] into [[DataFrame]].
   */
  implicit def dataFrameToTestFuncWrapper(df: DataFrame): TestFuncWrapper =
    new TestFuncWrapper(df)

  def sigmoid(exprs: Column*): Column = withExpr {
    planHiveGenericUDF(
      "hivemall.tools.math.SigmoidGenericUDF",
      "sigmoid",
      exprs
    )
  }

  /**
   * A convenient function to wrap an expression and produce a Column.
   */
  @inline private def withExpr(expr: Expression): Column = Column(expr)
}

class MiscBenchmark extends BenchmarkBase {

  val numIters = 10

  private def addBenchmarkCase(name: String, df: DataFrame)(implicit benchmark: Benchmark): Unit = {
    benchmark.addCase(name, numIters) {
      _ => df.queryExecution.executedPlan.execute().foreach(x => {})
    }
  }

  TestUtils.benchmark("closure/exprs/spark-udf/hive-udf") {
    /**
     * Java HotSpot(TM) 64-Bit Server VM 1.8.0_31-b13 on Mac OS X 10.10.2
     * Intel(R) Core(TM) i7-4578U CPU @ 3.00GHz
     *
     * sigmoid functions:       Best/Avg Time(ms)    Rate(M/s)   Per Row(ns)   Relative
     * --------------------------------------------------------------------------------
     * exprs                         7708 / 8173          3.4         294.0       1.0X
     * closure                       7722 / 8342          3.4         294.6       1.0X
     * spark-udf                     7963 / 8350          3.3         303.8       1.0X
     * hive-udf                    13977 / 14050          1.9         533.2       0.6X
     */
    import sparkSession.sqlContext.implicits._
    val N = 1L << 18
    val testDf = sparkSession.range(N).selectExpr("rand() AS value").cache

    // First, cache data
    testDf.count

    implicit val benchmark = new Benchmark("sigmoid", N)
    def sigmoidExprs(expr: Column): Column = {
      val one: () => Literal = () => Literal.create(1.0, DoubleType)
      Column(one()) / (Column(one()) + exp(-expr))
    }
    addBenchmarkCase(
      "exprs",
      testDf.select(sigmoidExprs($"value"))
    )
    implicit val encoder = RowEncoder(StructType(StructField("value", DoubleType) :: Nil))
    addBenchmarkCase(
      "closure",
      testDf.map { d =>
        Row(1.0 / (1.0 + Math.exp(-d.getDouble(0))))
      }
    )
    val sigmoidUdf = udf { (d: Double) => 1.0 / (1.0 + Math.exp(-d)) }
    addBenchmarkCase(
      "spark-udf",
      testDf.select(sigmoidUdf($"value"))
    )
    addBenchmarkCase(
      "hive-udf",
      testDf.select(TestFuncWrapper.sigmoid($"value"))
    )
    benchmark.run()
  }

  TestUtils.benchmark("top-k query") {
    /**
     * Java HotSpot(TM) 64-Bit Server VM 1.8.0_31-b13 on Mac OS X 10.10.2
     * Intel(R) Core(TM) i7-4578U CPU @ 3.00GHz
     *
     * top-k (k=100):          Best/Avg Time(ms)    Rate(M/s)   Per Row(ns)   Relative
     * -------------------------------------------------------------------------------
     * rank                       62748 / 62862          0.4        2393.6       1.0X
     * each_top_k (hive-udf)      41421 / 41736          0.6        1580.1       1.5X
     * each_top_k (exprs)         15793 / 16394          1.7         602.5       4.0X
     */
    import sparkSession.sqlContext.implicits._
    import TestFuncWrapper._
    val topK = 100
    val N = 1L << 20
    val numGroup = 3
    val testDf = sparkSession.range(N).selectExpr(
      s"id % $numGroup AS key", "rand() AS x", "CAST(id AS STRING) AS value"
    ).cache

    // First, cache data
    testDf.count

    implicit val benchmark = new Benchmark(s"top-k (k=$topK)", N)
    addBenchmarkCase(
      "rank",
      testDf.withColumn("rank", rank().over(Window.partitionBy($"key").orderBy($"x".desc)))
        .where($"rank" <= topK)
    )
    addBenchmarkCase(
      "each_top_k (hive-udf)",
      testDf.hive_each_top_k(lit(topK), $"key", $"x", $"key", $"value")
    )
    addBenchmarkCase(
      "each_top_k (exprs)",
      testDf.each_top_k(lit(topK), $"x".as("score"), $"key".as("group"))
    )
    benchmark.run()
  }

  TestUtils.benchmark("top-k join query") {
    /**
     * Java HotSpot(TM) 64-Bit Server VM 1.8.0_31-b13 on Mac OS X 10.10.2
     * Intel(R) Core(TM) i7-4578U CPU @ 3.00GHz
     *
     * top-k join (k=3):       Best/Avg Time(ms)    Rate(M/s)   Per Row(ns)   Relative
     * -------------------------------------------------------------------------------
     * join + rank                65959 / 71324          0.0      503223.9       1.0X
     * join + each_top_k          66093 / 78864          0.0      504247.3       1.0X
     * top_k_join                   5013 / 5431          0.0       38249.3      13.2X
     */
    import sparkSession.sqlContext.implicits._
    val topK = 3
    val N = 1L << 10
    val M = 1L << 10
    val numGroup = 3
    val inputDf = sparkSession.range(N).selectExpr(
      s"CAST(rand() * $numGroup AS INT) AS group", "id AS userId", "rand() AS x", "rand() AS y"
    ).cache
    val masterDf = sparkSession.range(M).selectExpr(
      s"id % $numGroup AS group", "id AS posId", "rand() AS x", "rand() AS y"
    ).cache

    // First, cache data
    inputDf.count
    masterDf.count

    implicit val benchmark = new Benchmark(s"top-k join (k=$topK)", N)
    // Define a score column
    val distance = sqrt(
      pow(inputDf("x") - masterDf("x"), lit(2.0)) +
      pow(inputDf("y") - masterDf("y"), lit(2.0))
    ).as("score")
    addBenchmarkCase(
      "join + rank",
      inputDf.join(masterDf, inputDf("group") === masterDf("group"))
        .select(inputDf("group"), $"userId", $"posId", distance)
        .withColumn(
          "rank", rank().over(Window.partitionBy($"group", $"userId").orderBy($"score".desc)))
        .where($"rank" <= topK)
    )
    addBenchmarkCase(
      "join + each_top_k",
      inputDf.join(masterDf, inputDf("group") === masterDf("group"))
        .each_top_k(lit(topK), distance, inputDf("group").as("group"))
    )
    addBenchmarkCase(
      "top_k_join",
      inputDf.top_k_join(lit(topK), masterDf, inputDf("group") === masterDf("group"), distance)
    )
    benchmark.run()
  }

  TestUtils.benchmark("codegen top-k join") {
    /**
     * Java HotSpot(TM) 64-Bit Server VM 1.8.0_31-b13 on Mac OS X 10.10.2
     * Intel(R) Core(TM) i7-4578U CPU @ 3.00GHz
     *
     * top_k_join:                 Best/Avg Time(ms)    Rate(M/s)   Per Row(ns)   Relative
     * -----------------------------------------------------------------------------------
     * top_k_join wholestage off           3 /    5       2751.9           0.4       1.0X
     * top_k_join wholestage on            1 /    1       6494.4           0.2       2.4X
     */
    val topK = 3
    val N = 1L << 23
    val M = 1L << 22
    val numGroup = 3
    val inputDf = sparkSession.range(N).selectExpr(
      s"CAST(rand() * $numGroup AS INT) AS group", "id AS userId", "rand() AS x", "rand() AS y"
    ).cache
    val masterDf = sparkSession.range(M).selectExpr(
      s"id % $numGroup AS group", "id AS posId", "rand() AS x", "rand() AS y"
    ).cache

    // First, cache data
    inputDf.count
    masterDf.count

    // Define a score column
    val distance = sqrt(
      pow(inputDf("x") - masterDf("x"), lit(2.0)) +
      pow(inputDf("y") - masterDf("y"), lit(2.0))
    )
    runBenchmark("top_k_join", N) {
      inputDf.top_k_join(lit(topK), masterDf, inputDf("group") === masterDf("group"),
        distance.as("score"))
    }
  }
}
