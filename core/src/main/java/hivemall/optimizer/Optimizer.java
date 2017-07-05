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
package hivemall.optimizer;

import hivemall.model.IWeightValue;
import hivemall.model.WeightValue;
import hivemall.utils.lang.Primitives;

import java.util.Map;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

public interface Optimizer {

    /**
     * Update the weights of models
     */
    float update(@Nonnull Object feature, float weight, float gradient);

    /**
     * Count up #step to tune learning rate
     */
    void proceedStep();

    @Nonnull
    String getOptimizerName();

    @NotThreadSafe
    static abstract class OptimizerBase implements Optimizer {

        @Nonnull
        protected final EtaEstimator _eta;
        @Nonnull
        protected final Regularization _reg;
        @Nonnegative
        protected int _numStep = 1;

        public OptimizerBase(@Nonnull Map<String, String> options) {
            this._eta = EtaEstimator.get(options);
            this._reg = Regularization.get(options);
        }

        @Override
        public void proceedStep() {
            _numStep++;
        }

        /**
         * Update the given weight by the given gradient.
         * 
         * @return new weight to be set
         */
        protected float update(@Nonnull final IWeightValue weight, final float gradient) {
            float oldWeight = weight.get();
            float g = _reg.regularize(oldWeight, gradient);
            float delta = computeDelta(weight, g);
            float newWeight = oldWeight - _eta.eta(_numStep) * delta;
            weight.set(newWeight);
            return newWeight;
        }

        /**
         * Compute a delta to update
         */
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            return gradient;
        }

    }

    static final class SGD extends OptimizerBase {

        private final IWeightValue weightValueReused;

        public SGD(final Map<String, String> options) {
            super(options);
            this.weightValueReused = new WeightValue(0.f);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
            weightValueReused.set(weight);
            update(weightValueReused, gradient);
            return weightValueReused.get();
        }

        @Override
        public String getOptimizerName() {
            return "sgd";
        }

    }

    static abstract class AdaGrad extends OptimizerBase {

        private final float eps;
        private final float scale;

        public AdaGrad(Map<String, String> options) {
            super(options);
            this.eps = Primitives.parseFloat(options.get("eps"), 1.0f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float new_scaled_sum_sqgrad = weight.getSumOfSquaredGradients() + gradient
                    * (gradient / scale);
            weight.setSumOfSquaredGradients(new_scaled_sum_sqgrad);
            return gradient / ((float) Math.sqrt(new_scaled_sum_sqgrad * scale) + eps);
        }

        @Override
        public String getOptimizerName() {
            return "adagrad";
        }

    }

    static abstract class AdaDelta extends OptimizerBase {

        private final float decay;
        private final float eps;
        private final float scale;

        public AdaDelta(Map<String, String> options) {
            super(options);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.95f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1e-6f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_sum_sqgrad = weight.getSumOfSquaredGradients();
            float old_sum_squared_delta_x = weight.getSumOfSquaredDeltaX();
            float new_scaled_sum_sqgrad = (decay * old_scaled_sum_sqgrad)
                    + ((1.f - decay) * gradient * (gradient / scale));
            float delta = (float) Math.sqrt((old_sum_squared_delta_x + eps)
                    / (new_scaled_sum_sqgrad * scale + eps))
                    * gradient;
            float new_sum_squared_delta_x = (decay * old_sum_squared_delta_x)
                    + ((1.f - decay) * delta * delta);
            weight.setSumOfSquaredGradients(new_scaled_sum_sqgrad);
            weight.setSumOfSquaredDeltaX(new_sum_squared_delta_x);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "adadelta";
        }

    }

    /**
     * Adam, an algorithm for first-order gradient-based optimization of stochastic objective
     * functions, based on adaptive estimates of lower-order moments.
     *
     * - D. P. Kingma and J. L. Ba: "ADAM: A Method for Stochastic Optimization." arXiv preprint
     * arXiv:1412.6980v8, 2014.
     */
    static abstract class Adam extends OptimizerBase {

        private final float beta;
        private final float gamma;
        private final float eps_hat;

        public Adam(Map<String, String> options) {
            super(options);
            this.beta = Primitives.parseFloat(options.get("beta"), 0.9f);
            this.gamma = Primitives.parseFloat(options.get("gamma"), 0.999f);
            this.eps_hat = Primitives.parseFloat(options.get("eps_hat"), 1e-8f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float val_m = beta * weight.getM() + (1.f - beta) * gradient;
            float val_v = gamma * weight.getV() + (float) ((1.f - gamma) * Math.pow(gradient, 2.0));
            float val_m_hat = val_m / (float) (1.f - Math.pow(beta, _numStep));
            float val_v_hat = val_v / (float) (1.f - Math.pow(gamma, _numStep));
            float delta = val_m_hat / (float) (Math.sqrt(val_v_hat) + eps_hat);
            weight.setM(val_m);
            weight.setV(val_v);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "adam";
        }

    }

    static abstract class AdagradRDA extends OptimizerBase {

        @Nonnull
        private final AdaGrad optimizerImpl;
        private final float lambda;

        public AdagradRDA(@Nonnull AdaGrad optimizerImpl, @Nonnull Map<String, String> options) {
            super(options);
            this.optimizerImpl = optimizerImpl;
            this.lambda = Primitives.parseFloat(options.get("lambda"), 1e-6f);
        }

        @Override
        protected float update(@Nonnull final IWeightValue weight, final float gradient) {
            final float new_sum_grad = weight.getSumOfGradients() + gradient;
            // sign(u_{t,i})
            final float sign = (new_sum_grad > 0.f) ? 1.f : -1.f;
            // |u_{t,i}|/t - \lambda
            final float meansOfGradients = (sign * new_sum_grad / _numStep) - lambda;
            if (meansOfGradients < 0.f) {
                // x_{t,i} = 0
                weight.set(0.f);
                weight.setSumOfSquaredGradients(0.f);
                weight.setSumOfGradients(0.f);
                return 0.f;
            } else {
                // x_{t,i} = -sign(u_{t,i}) * \frac{\eta t}{\sqrt{G_{t,ii}}}(|u_{t,i}|/t - \lambda)
                float newWeight = -1.f * sign * _eta.eta(_numStep) * _numStep
                        * optimizerImpl.computeDelta(weight, meansOfGradients);
                weight.set(newWeight);
                weight.setSumOfGradients(new_sum_grad);
                return newWeight;
            }
        }

        @Override
        public String getOptimizerName() {
            return "adagrad_rda";
        }

    }

}
