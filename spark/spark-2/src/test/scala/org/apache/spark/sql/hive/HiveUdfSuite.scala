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

import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HivemallUtils._
import org.apache.spark.test.HivemallFeatureQueryTest
import org.apache.spark.test.VectorQueryTest

final class HiveUdfWithFeatureSuite extends HivemallFeatureQueryTest {
  import hiveContext.implicits._
  import hiveContext._

  test("hivemall_version") {
    sql(s"""
         | CREATE TEMPORARY FUNCTION hivemall_version
         |   AS '${classOf[hivemall.HivemallVersionUDF].getName}'
       """.stripMargin)

    checkAnswer(
      sql(s"SELECT DISTINCT hivemall_version()"),
      Row("0.4.2-rc.2")
    )

    // sql("DROP TEMPORARY FUNCTION IF EXISTS hivemall_version")
    // reset()
  }

  test("train_logregr") {
    TinyTrainData.createOrReplaceTempView("TinyTrainData")
    sql(s"""
         | CREATE TEMPORARY FUNCTION train_logregr
         |   AS '${classOf[hivemall.regression.LogressUDTF].getName}'
       """.stripMargin)
    sql(s"""
         | CREATE TEMPORARY FUNCTION add_bias
         |   AS '${classOf[hivemall.ftvec.AddBiasUDFWrapper].getName}'
       """.stripMargin)

    val model = sql(
      s"""
         | SELECT feature, AVG(weight) AS weight
         |   FROM (
         |       SELECT train_logregr(add_bias(features), label) AS (feature, weight)
         |         FROM TinyTrainData
         |     ) t
         |   GROUP BY feature
       """.stripMargin)

    checkAnswer(
      model.select($"feature"),
      Seq(Row("0"), Row("1"), Row("2"))
    )

    // TODO: Why 'train_logregr' is not registered in HiveMetaStore?
    // ERROR RetryingHMSHandler: MetaException(message:NoSuchObjectException
    //   (message:Function default.train_logregr does not exist))
    //
    // hiveContext.sql("DROP TEMPORARY FUNCTION IF EXISTS train_logregr")
    // hiveContext.reset()
  }

  test("each_top_k") {
    val testDf = Seq(
      ("a", "1", 0.5, Array(0, 1, 2)),
      ("b", "5", 0.1, Array(3)),
      ("a", "3", 0.8, Array(2, 5)),
      ("c", "6", 0.3, Array(1, 3)),
      ("b", "4", 0.3, Array(2)),
      ("a", "2", 0.6, Array(1))
    ).toDF("key", "value", "score", "data")

    import testDf.sqlContext.implicits._
    testDf.repartition($"key").sortWithinPartitions($"key").createOrReplaceTempView("TestData")
    sql(s"""
         | CREATE TEMPORARY FUNCTION each_top_k
         |   AS '${classOf[hivemall.tools.EachTopKUDTF].getName}'
       """.stripMargin)

    // Compute top-1 rows for each group
    checkAnswer(
      sql("SELECT each_top_k(1, key, score, key, value) FROM TestData"),
      Row(1, 0.8, "a", "3") ::
      Row(1, 0.3, "b", "4") ::
      Row(1, 0.3, "c", "6") ::
      Nil
    )

    // Compute reverse top-1 rows for each group
    checkAnswer(
      sql("SELECT each_top_k(-1, key, score, key, value) FROM TestData"),
      Row(1, 0.5, "a", "1") ::
      Row(1, 0.1, "b", "5") ::
      Row(1, 0.3, "c", "6") ::
      Nil
    )
  }
}

final class HiveUdfWithVectorSuite extends VectorQueryTest {
  import hiveContext._

  test("to_hivemall_features") {
    mllibTrainDf.createOrReplaceTempView("mllibTrainDf")
    hiveContext.udf.register("to_hivemall_features", _to_hivemall_features)
    checkAnswer(
      sql(
        s"""
           | SELECT to_hivemall_features(features)
           |   FROM mllibTrainDf
         """.stripMargin),
      Seq(
        Row(Seq("0:1.0", "2:2.0", "4:3.0")),
        Row(Seq("0:1.0", "3:1.5", "4:2.1", "6:1.2")),
        Row(Seq("0:1.1", "3:1.0", "4:2.3", "6:1.0")),
        Row(Seq("1:4.0", "3:5.0", "5:6.0"))
      )
    )
  }

  ignore("append_bias") {
    mllibTrainDf.createOrReplaceTempView("mllibTrainDf")
    hiveContext.udf.register("append_bias", _append_bias)
    hiveContext.udf.register("to_hivemall_features", _to_hivemall_features)
    /**
     * TODO: This test throws an exception:
     * Failed to analyze query: org.apache.spark.sql.AnalysisException: cannot resolve
     *   'UDF(UDF(features))' due to data type mismatch: argument 1 requires vector type,
     *    however, 'UDF(features)' is of vector type.; line 2 pos 8
     */
    checkAnswer(
      sql(
        s"""
           | SELECT to_hivemall_features(append_bias(features))
           |   FROM mllibTrainDF
         """.stripMargin),
       Seq(
        Row(Seq("0:1.0", "0:1.0", "2:2.0", "4:3.0")),
        Row(Seq("0:1.0", "0:1.0", "3:1.5", "4:2.1", "6:1.2")),
        Row(Seq("0:1.0", "0:1.1", "3:1.0", "4:2.3", "6:1.0")),
        Row(Seq("0:1.0", "1:4.0", "3:5.0", "5:6.0"))
      )
    )
  }

  ignore("explode_vector") {
    // TODO: Spark-2.0 does not support use-defined generator function in
    // `org.apache.spark.sql.UDFRegistration`.
  }
}
