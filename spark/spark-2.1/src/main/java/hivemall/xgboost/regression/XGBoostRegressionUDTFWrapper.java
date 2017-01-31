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
package hivemall.xgboost.regression;

import java.util.UUID;

import org.apache.hadoop.hive.ql.exec.Description;

/** An alternative implementation of [[hivemall.xgboost.regression.XGBoostRegressionUDTF]]. */
@Description(
    name = "train_xgboost_regr",
    value = "_FUNC_(string[] features, double target [, string options]) - Returns a relation consisting of <string model_id, array<byte> pred_model>"
)
public class XGBoostRegressionUDTFWrapper extends XGBoostRegressionUDTF {
    private long sequence;
    private long taskId;

    public XGBoostRegressionUDTFWrapper() {
        this.sequence = 0L;
        this.taskId = Thread.currentThread().getId();
    }

    @Override
    protected String generateUniqueModelId() {
        sequence++;
        /**
         * TODO: Check if it is unique over all tasks in executors of Spark.
         */
        return "xgbmodel-" + taskId + "-" + UUID.randomUUID() + "-" + sequence;
    }
}
