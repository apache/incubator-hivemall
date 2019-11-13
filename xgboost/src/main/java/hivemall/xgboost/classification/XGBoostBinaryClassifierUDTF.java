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
package hivemall.xgboost.classification;

import hivemall.xgboost.XGBoostTrainUDTF;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;

/**
 * A XGBoost binary classifier.
 */
@Description(name = "train_xgboost_classifier",
        value = "_FUNC_(array<string> features, double target [, string options])"
                + " - Returns a relation consisting of <string model_id, array<byte> pred_model>")
public final class XGBoostBinaryClassifierUDTF extends XGBoostTrainUDTF {

    public XGBoostBinaryClassifierUDTF() {
        super();
    }

    {
        params.put("objective", "binary:logistic");
    }

    @Override
    protected float processTargetValue(final float target) throws HiveException {
        if (target != -1 && target != 0 && target != 1) {
            throw new UDFArgumentException("Invalid label value for classification: " + target);
        }
        return target > 0.f ? 1.f : 0.f;
    }

}
