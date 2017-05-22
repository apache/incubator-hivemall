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
package hivemall.classifier;

import hivemall.annotations.Since;
import hivemall.annotations.VisibleForTesting;
import hivemall.model.FeatureValue;
import hivemall.optimizer.LossFunctions;
import hivemall.optimizer.LossFunctions.LossFunction;
import hivemall.optimizer.LossFunctions.LossType;
import hivemall.optimizer.Optimizer;
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
 * A general classifier class that can select a loss function and an optimization function.
 */
@Description(name = "train_classifier",
        value = "_FUNC_(list<string|int|bigint> features, int label [, const string options])"
                + " - Returns a relation consists of <string|int|bigint feature, float weight>",
        extended = "Build a prediction model by a generic classifier")
@Since(version = "0.5-rc.1")
public final class GeneralClassifierUDTF extends BinaryOnlineClassifierUDTF {

    private Optimizer optimizer;
    @Nonnull
    private final Map<String, String> optimizerOptions;
    private LossFunction lossFunction;

    private float loss;

    public GeneralClassifierUDTF() {
        super(true);
        this.optimizerOptions = OptimizerOptions.create();
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(
                "_FUNC_ takes 2 or 3 arguments: List<Text|Int|BitInt> features, int label "
                        + "[, constant string options]");
        }

        StructObjectInspector outputOI = super.initialize(argOIs);

        try {
            this.optimizer = createOptimizer(optimizerOptions);
        } catch (Throwable e) {
            throw new UDFArgumentException(e.getMessage());
        }

        if ((optimizer instanceof Optimizer.OptimizerBase.AdagradRDA) && is_mini_batch) {
            throw new UDFArgumentException("Currently `-mini_batch` option is NOT available for AdaGradRDA");
        }

        this.loss = 0.f;

        return outputOI;
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("loss", "loss_function", true,
            "Loss function [default: HingeLoss, LogLoss, SquaredHingeLoss, ModifiedHuberLoss, "
                    + "SquaredLoss, QuantileLoss, EpsilonInsensitiveLoss, HuberLoss]");
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
            this.lossFunction = LossFunctions.getLossFunction(LossType.HingeLoss);
        }
        OptimizerOptions.propcessOptions(cl, optimizerOptions);
        return cl;
    }

    @Override
    protected void train(@Nonnull final FeatureValue[] features, final int label) {
        float predicted = predict(features);
        float y = label > 0 ? 1.f : -1.f;
        update(features, y, predicted);
    }

    @Override
    protected void update(@Nonnull final FeatureValue[] features, final float label,
            final float predicted) {
        float dloss = lossFunction.dloss(predicted, label);
        if (is_mini_batch) {
            // for mini-batch, consider mean of accumulated loss
            this.loss += lossFunction.loss(predicted, label);

            accumulateUpdate(features, dloss);

            if (sampled >= mini_batch_size) {
                batchUpdate();
                this.loss = 0.f;
            }
        } else {
            this.loss = lossFunction.loss(predicted, label);
            onlineUpdate(features, dloss);
        }
        optimizer.proceedStep();
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
    }

    @VisibleForTesting
    float getLoss() {
        if (is_mini_batch) {
            if (sampled == 0) {
                return 0.f;
            } else {
                return loss / sampled;
            }
        }
        return loss;
    }

}
