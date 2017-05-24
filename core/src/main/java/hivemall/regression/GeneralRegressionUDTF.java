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
import hivemall.annotations.VisibleForTesting;
import hivemall.model.FeatureValue;
import hivemall.optimizer.LossFunctions;
import hivemall.optimizer.LossFunctions.LossFunction;
import hivemall.optimizer.LossFunctions.LossType;
import hivemall.optimizer.Optimizer;
import hivemall.optimizer.Optimizer.OptimizerBase;
import hivemall.optimizer.OptimizerOptions;
import hivemall.utils.lang.FloatAccumulator;

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
    private LossFunction lossFunction;

    private float cumLoss;

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

        if (lossFunction.forBinaryClassification()) {
            throw new UDFArgumentException("The loss function `" + lossFunction.getType()
                    + "` is not designed for regression");

        }

        try {
            this.optimizer = createOptimizer(optimizerOptions);
        } catch (Throwable e) {
            throw new UDFArgumentException(e.getMessage());
        }

        if ((optimizer instanceof OptimizerBase.AdagradRDA) && is_mini_batch) {
            throw new UDFArgumentException("Currently `-mini_batch` option is NOT available for AdaGradRDA");
        }

        this.cumLoss = 0.f;

        return outputOI;
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("loss", "loss_function", true,
            "Loss function [default: SquaredLoss, QuantileLoss, EpsilonInsensitiveLoss, HuberLoss]");
        OptimizerOptions.setup(opts);
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);
        if (cl.hasOption("loss_function")) {
            try {
                this.lossFunction = LossFunctions.getLossFunction(cl.getOptionValue("loss_function"));
            } catch (Throwable e) {
                throw new UDFArgumentException(e.getMessage());
            }
        } else {
            this.lossFunction = LossFunctions.getLossFunction(LossType.SquaredLoss);
        }
        OptimizerOptions.propcessOptions(cl, optimizerOptions);
        return cl;
    }

    @Override
    protected void update(@Nonnull final FeatureValue[] features, final float target,
            final float predicted) {
        this.cumLoss += lossFunction.loss(predicted, target); // retain cumulative loss to check convergence
        float dloss = lossFunction.dloss(predicted, target);
        if (is_mini_batch) {
            accumulateUpdate(features, dloss);
            if (sampled >= mini_batch_size) {
                batchUpdate();
            }
        } else {
            onlineUpdate(features, dloss);
        }
    }

    @Override
    protected void accumulateUpdate(@Nonnull final FeatureValue[] features, final float dloss) {
        for (FeatureValue f : features) {
            Object feature = f.getFeature();
            float xi = f.getValueAsFloat();
            float weight = model.getWeight(feature);

            // compute new weight, but still not set to the model
            float new_weight = optimizer.update(feature, weight, dloss * xi);

            // (w_i - eta * delta_1) + (w_i - eta * delta_2) + ... + (w_i - eta * delta_M)
            FloatAccumulator acc = accumulated.get(feature);
            if (acc == null) {
                acc = new FloatAccumulator(new_weight);
                accumulated.put(feature, acc);
            } else {
                acc.add(new_weight);
            }
        }
        sampled++;
    }

    @Override
    protected void batchUpdate() {
        if (accumulated.isEmpty()) {
            this.sampled = 0;
            return;
        }

        for (Map.Entry<Object, FloatAccumulator> e : accumulated.entrySet()) {
            Object feature = e.getKey();
            FloatAccumulator v = e.getValue();
            float new_weight = v.get(); // w_i - (eta / M) * (delta_1 + delta_2 + ... + delta_M)
            model.setWeight(feature, new_weight);
        }

        optimizer.proceedStep();

        accumulated.clear();
        this.sampled = 0;
    }

    @Override
    protected void onlineUpdate(@Nonnull final FeatureValue[] features, float dloss) {
        for (FeatureValue f : features) {
            Object feature = f.getFeature();
            float xi = f.getValueAsFloat();
            float weight = model.getWeight(feature);
            float new_weight = optimizer.update(feature, weight, dloss * xi);
            model.setWeight(feature, new_weight);
        }
        optimizer.proceedStep();
    }

    @VisibleForTesting
    float getCumulativeLoss() {
        return cumLoss;
    }

    @VisibleForTesting
    void resetCumulativeLoss() {
        this.cumLoss = 0.f;
    }

}
