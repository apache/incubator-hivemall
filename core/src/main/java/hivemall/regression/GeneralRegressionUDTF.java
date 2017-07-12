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

import hivemall.GeneralLearnerBaseUDTF;
import hivemall.annotations.Since;
import hivemall.model.FeatureValue;
import hivemall.optimizer.LossFunctions.LossFunction;
import hivemall.optimizer.LossFunctions.LossType;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;

/**
 * A general regression class with replaceable optimization functions.
 */
@Description(name = "train_regression",
        value = "_FUNC_(list<string|int|bigint> features, double label [, const string options])"
                + " - Returns a relation consists of <string|int|bigint feature, float weight>",
        extended = "Build a prediction model by a generic regressor")
@Since(version = "0.5-rc.1")
public final class GeneralRegressionUDTF extends GeneralLearnerBaseUDTF {

    @Override
    protected String getLossOptionDescription() {
        return "Loss function [SquaredLoss (default), QuantileLoss, EpsilonInsensitiveLoss, "
                + "SquaredEpsilonInsensitiveLoss, HuberLoss]";
    }

    @Override
    protected LossType getDefaultLossType() {
        return LossType.SquaredLoss;
    }

    @Override
    protected void checkLossFunction(@Nonnull LossFunction lossFunction)
            throws UDFArgumentException {
        if (!lossFunction.forRegression()) {
            throw new UDFArgumentException("The loss function `" + lossFunction.getType()
                    + "` is not designed for regression");
        }
    }

    @Override
    protected void checkTargetValue(float label) throws UDFArgumentException {}

    @Override
    protected void train(@Nonnull final FeatureValue[] features, final float target) {
        float p = predict(features);
        update(features, target, p);
    }

}
