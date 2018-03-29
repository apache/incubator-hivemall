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

import scala.language.implicitConversions

import org.apache.spark.internal.config.{ConfigBuilder, ConfigEntry, ConfigReader}
import org.apache.spark.sql.internal.SQLConf

object HivemallConf {

  /**
   * Implicitly injects the [[HivemallConf]] into [[SQLConf]].
   */
  implicit def SQLConfToHivemallConf(conf: SQLConf): HivemallConf = new HivemallConf(conf)

  private val sqlConfEntries = java.util.Collections.synchronizedMap(
    new java.util.HashMap[String, ConfigEntry[_]]())

  private def register(entry: ConfigEntry[_]): Unit = sqlConfEntries.synchronized {
    require(!sqlConfEntries.containsKey(entry.key),
      s"Duplicate SQLConfigEntry. ${entry.key} has been registered")
    sqlConfEntries.put(entry.key, entry)
  }

  // For testing only
  // TODO: Need to add tests for the configurations
  private[sql] def unregister(entry: ConfigEntry[_]): Unit = sqlConfEntries.synchronized {
    sqlConfEntries.remove(entry.key)
  }

  def buildConf(key: String): ConfigBuilder = ConfigBuilder(key).onCreate(register)

  val FEATURE_SELECTION_ENABLED =
    buildConf("spark.sql.optimizer.featureSelection.enabled")
    .doc("Whether feature selections are applied in the optimizer")
    .booleanConf
    .createWithDefault(false)

  val FEATURE_SELECTION_VARIANCE_THRESHOLD =
    buildConf("spark.sql.optimizer.featureSelection.varianceThreshold")
    .doc("Specifies the threshold of variances to filter out features")
    .doubleConf
    .createWithDefault(0.05)
}

class HivemallConf(conf: SQLConf) {
  import HivemallConf._

  private val reader = new ConfigReader(conf.settings)

  def featureSelectionEnabled: Boolean = getConf(FEATURE_SELECTION_ENABLED)

  def featureSelectionVarianceThreshold: Double = getConf(FEATURE_SELECTION_VARIANCE_THRESHOLD)

  /** ********************** SQLConf functionality methods ************ */

  /**
   * Return the value of Hivemall configuration property for the given key. If the key is not set
   * yet, return `defaultValue` in [[ConfigEntry]].
   */
  private def getConf[T](entry: ConfigEntry[T]): T = {
    require(sqlConfEntries.get(entry.key) == entry, s"$entry is not registered")
    entry.readFrom(reader)
  }
}
