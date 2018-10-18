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

import org.apache.spark.sql.{QueryTest, Row}
import org.apache.spark.sql.hive.test.TestHiveSingleton
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.optimizer.VarianceThreshold
import org.apache.spark.sql.test.SQLTestUtils
import org.apache.spark.sql.types._


class FeatureSelectionRuleSuite extends SQLTestUtils with TestHiveSingleton {

  import hiveContext.implicits._

  protected override def beforeAll(): Unit = {
    super.beforeAll()
    // Sets user-defined optimization rules for feature selection
    hiveContext.experimental.extraOptimizations = Seq(
      new VarianceThreshold(hiveContext.conf))
  }

  test("filter out features with low variances") {
    withSQLConf(
        HivemallConf.FEATURE_SELECTION_ENABLED.key -> "true",
        HivemallConf.FEATURE_SELECTION_VARIANCE_THRESHOLD.key -> "0.1",
        SQLConf.CBO_ENABLED.key -> "true",
        SQLConf.HISTOGRAM_ENABLED.key -> "true") {
      withTable("t") {
        withTempDir { dir =>
          Seq((1, "one", 1.0, 1.0),
              (1, "two", 1.1, 2.3),
              (1, "three", 0.9, 3.5),
              (1, "one", 0.9, 10.3))
            .toDF("c0", "c1", "c2", "c3")
            .write
            .parquet(s"${dir.getAbsolutePath}/t")

          spark.read.parquet(s"${dir.getAbsolutePath}/t").write.saveAsTable("t")

          sql("ANALYZE TABLE t COMPUTE STATISTICS FOR COLUMNS c0, c1, c2, c3")

          // Filters out `c0` and `c2` because of low variances
          val optimizedPlan = sql("SELECT c0, * FROM t").queryExecution.optimizedPlan
          assert(optimizedPlan.output.map(_.name) === Seq("c1", "c3"))
        }
      }
    }
  }
}
