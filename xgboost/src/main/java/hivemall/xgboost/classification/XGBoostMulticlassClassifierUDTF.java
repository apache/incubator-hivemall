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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;

/**
 * A XGBoost multiclass classifier.
 */
@Description(name = "train_multiclass_xgboost_classifier",
        value = "_FUNC_(string[] features, double target [, string options]) - Returns a relation consisting of <string model_id, array<byte> pred_model>")
public final class XGBoostMulticlassClassifierUDTF extends XGBoostTrainUDTF {

    private int numClass;

    public XGBoostMulticlassClassifierUDTF() {
        super();
    }

    {
        params.put("objective", "multi:softprob");
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("num_class", true, "Number of classes to classify");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        if (!cl.hasOption("num_class")) {
            throw new UDFArgumentException("-num_class is required for multiclass classification");
        }

        this.numClass = Integer.parseInt(cl.getOptionValue("num_class"));
        params.put("num_class", numClass);

        return cl;
    }

    @Override
    protected float processTargetValue(final float target) throws HiveException {
        final int clazz = (int) target;
        if (clazz != target) {
            throw new UDFArgumentException("Invalid target value for class label: " + target);
        }
        if (clazz < 0 || clazz >= numClass) {
            throw new UDFArgumentException("target must be {0.0, ..., "
                    + String.format("%.1f", (numClass - 1.0)) + "}: " + target);
        }
        return target;
    }

}
