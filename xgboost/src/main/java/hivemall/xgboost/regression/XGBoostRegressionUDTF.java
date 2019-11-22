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

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.metadata.HiveException;

import hivemall.xgboost.XGBoostBaseUDTF;

/**
 * A XGBoost regressor.
 */
@Description(name = "train_xgboost_regr",
        value = "_FUNC_(string[] features, double target [, string options]) - Returns a relation consisting of <string model_id, array<byte> pred_model>")
public final class XGBoostRegressionUDTF extends XGBoostBaseUDTF {

    public XGBoostRegressionUDTF() {
        super();
    }

    {
        // Settings for logistic regression
        params.put("objective", "reg:logistic");
        params.put("eval_metric", "rmse");
    }

    @Override
    protected void checkTargetValue(final float target) throws HiveException {
        if (target < 0.f || target > 1.f) {
            throw new HiveException("target must be in range 0 to 1: " + target);
        }
    }

}
