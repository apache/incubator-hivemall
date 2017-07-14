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

import hivemall.GeneralLearnerBaseUDTF;
import hivemall.annotations.Since;
import hivemall.model.FeatureValue;
import hivemall.optimizer.LossFunctions.LossFunction;
import hivemall.optimizer.LossFunctions.LossType;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;

/**
 * A general classifier class that can select a loss function and an optimization function.
 */
@Description(name = "train_classifier",
        value = "_FUNC_(list<string|int|bigint> features, int label [, const string options])"
                + " - Returns a relation consists of <string|int|bigint feature, float weight>",
        extended = "Build a prediction model by a generic classifier")
@Since(version = "0.5-rc.1")
public final class GeneralClassifierUDTF extends GeneralLearnerBaseUDTF {

    @Override
    protected String getLossOptionDescription() {
        return "Loss function [HingeLoss (default), LogLoss, SquaredHingeLoss, ModifiedHuberLoss, or\n"
                + "a regression loss: SquaredLoss, QuantileLoss, EpsilonInsensitiveLoss, "
                + "SquaredEpsilonInsensitiveLoss, HuberLoss]";
    }

    @Override
    protected LossType getDefaultLossType() {
        return LossType.HingeLoss;
    }

    @Override
    protected void checkLossFunction(@Nonnull LossFunction lossFunction)
            throws UDFArgumentException {
        // will accepts both binary loss and regression loss functions
    }

    @Override
    protected void checkTargetValue(final float label) throws UDFArgumentException {
        if (label != -1 && label != 0 && label != 1) {
            throw new UDFArgumentException("Invalid label value for classification:  + label");
        }
    }

    @Override
    protected void train(@Nonnull final FeatureValue[] features, final float label) {
        float predicted = predict(features);
        float y = label > 0.f ? 1.f : -1.f;
        update(features, y, predicted);
    }

}
