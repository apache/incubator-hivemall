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

package org.apache.spark.sql.optimizer

import org.apache.spark.sql.catalyst.plans.logical.{Histogram, LogicalPlan, Project, Statistics}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf


/**
 * This optimizer rule removes features with low variance; it removes all features whose
 * variance doesn't meet some threshold. You can control this threshold by
 * `spark.sql.optimizer.featureSelection.varianceThreshold` (0.05 by default).
 */
class VarianceThreshold(conf: SQLConf) extends Rule[LogicalPlan] {
  import org.apache.spark.sql.hive.HivemallConf._

  private def featureSelectionEnabled: Boolean = conf.featureSelectionEnabled
  private def varianceThreshold: Double = conf.featureSelectionVarianceThreshold

  private def hasColumnHistogram(s: Statistics): Boolean = {
    s.attributeStats.exists { case (_, stat) =>
      stat.histogram.isDefined
    }
  }

  private def satisfyVarianceThreshold(histgramOption: Option[Histogram]): Boolean = {
    // TODO: Since binary types are not supported in histograms but they could frequently appear
    // in user schemas, we would be better to handle the case here.
    histgramOption.forall { hist =>
      // TODO: Make the value more precise by using `HistogramBin.ndv`
      val dataSeq = hist.bins.map { bin => (bin.hi + bin.lo) / 2 }
      val avg = dataSeq.sum / dataSeq.length
      val variance = dataSeq.map { d => Math.pow(avg - d, 2.0) }.sum / dataSeq.length
      varianceThreshold < variance
    }
  }

  override def apply(plan: LogicalPlan): LogicalPlan = plan match {
    case p if featureSelectionEnabled && hasColumnHistogram(p.stats) =>
      val attributeStats = p.stats.attributeStats
      val outputAttrs = p.output
      val projectList = outputAttrs.zip(outputAttrs.map { a => attributeStats.get(a)}).flatMap {
        case (_, Some(stat)) if !satisfyVarianceThreshold(stat.histogram) => None
        case (attr, _) => Some(attr)
      }
      if (projectList != outputAttrs) {
        Project(projectList, p)
      } else {
        p
      }
    case p => p
  }
}
