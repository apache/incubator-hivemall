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
package hivemall.regression;

import hivemall.annotations.Since;
import hivemall.model.FeatureValue;
import hivemall.optimizer.LossFunctions;
import hivemall.optimizer.Optimizer;
import hivemall.optimizer.OptimizerOptions;

import java.util.Map;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;

/**
 * A general regression class with replaceable optimization functions.
 */
@Description(name = "train_regression",
        value = "_FUNC_(list<string|int|bigint> features, double label [, const string options])"
                + " - Returns a relation consists of <string|int|bigint feature, float weight>",
        extended = "Build a prediction model by a generic regressor")
@Since(version = "0.5-rc.1")
public final class GeneralRegressionUDTF extends RegressionBaseUDTF {

    @Nonnull
    private final Map<String, String> optimizerOptions;
    private Optimizer optimizer;

    public GeneralRegressionUDTF() {
        super(true); // This enables new model interfaces
        this.optimizerOptions = OptimizerOptions.create();
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(this.getClass().getSimpleName()
                    + " takes 2 or 3 arguments: List<Text|Int|BitInt> features, float target "
                    + "[, constant string options]");
        }

        StructObjectInspector outputOI = super.initialize(argOIs);

        if (is_mini_batch) {
            throw new UDFArgumentException("_FUNC_ does not currently support `-mini_batch` option");
        }

        this.optimizer = createOptimizer(optimizerOptions);
        return outputOI;
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        OptimizerOptions.setup(opts);
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);
        OptimizerOptions.propcessOptions(cl, optimizerOptions);
        return cl;
    }

    @Override
    protected final void checkTargetValue(final float target) throws UDFArgumentException {
        if (target < 0.f || target > 1.f) {
            throw new UDFArgumentException("target must be in range 0 to 1: " + target);
        }
    }

    @Override
    protected void update(@Nonnull final FeatureValue[] features, final float target,
            final float predicted) {
        float loss = LossFunctions.logisticLoss(target, predicted);
        for (FeatureValue f : features) {
            Object feature = f.getFeature();
            float xi = f.getValueAsFloat();
            float weight = model.getWeight(feature);
            float new_weight = optimizer.update(feature, weight, -loss * xi);
            model.setWeight(feature, new_weight);
        }
        optimizer.proceedStep();
    }

}
