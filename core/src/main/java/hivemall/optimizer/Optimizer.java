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
import hivemall.utils.math.MathUtils;

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
        protected long _numStep = 1L;

        public OptimizerBase(@Nonnull Map<String, String> options) {
            this._eta = getEtaEstimator(options);
            this._reg = Regularization.get(options);
        }

        @Nonnull
        protected EtaEstimator getEtaEstimator(@Nonnull Map<String, String> options) {
            return EtaEstimator.get(options);
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
            float delta = computeDelta(weight, gradient);
            float eta = eta(_numStep);
            float reg = _reg.regularize(oldWeight, delta);
            float newWeight = oldWeight - eta * reg;
            weight.set(newWeight);
            return newWeight;
        }

        /**
         * @param t timestep
         * @return learning rate
         */
        protected float eta(final long t) {
            return _eta.eta(_numStep);
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

        public SGD(@Nonnull Map<String, String> options) {
            super(options);
            this.weightValueReused = new WeightValue(0.f);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
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

        public AdaGrad(@Nonnull Map<String, String> options) {
            super(options);
            this.eps = Primitives.parseFloat(options.get("eps"), 1.0f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_gg = weight.getSumOfSquaredGradients();
            float new_scaled_gg = old_scaled_gg + gradient * (gradient / scale);
            weight.setSumOfSquaredGradients(new_scaled_gg);
            return (float) (gradient / Math.sqrt(eps + ((double) old_scaled_gg) * scale));
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

        public AdaDelta(@Nonnull Map<String, String> options) {
            super(options);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.95f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1e-6f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected final EtaEstimator getEtaEstimator(@Nonnull Map<String, String> options) {
            // override default learning rate scheme
            if (!options.containsKey("eta")) {
                options.put("eta", "fixed");
            }
            if (!options.containsKey("eta0")) {
                options.put("eta0", "1.0");
            }
            return super.getEtaEstimator(options);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_sum_sqgrad = weight.getSumOfSquaredGradients();
            float old_sum_squared_delta_x = weight.getSumOfSquaredDeltaX();
            float new_scaled_sum_sqgrad = (decay * old_scaled_sum_sqgrad)
                    + ((1.f - decay) * gradient * (gradient / scale));
            float delta = (float) Math.sqrt(
                (old_sum_squared_delta_x + eps) / ((double) new_scaled_sum_sqgrad * scale + eps))
                    * gradient;
            float new_sum_squared_delta_x =
                    (decay * old_sum_squared_delta_x) + ((1.f - decay) * delta * delta);
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
     * - D. P. Kingma and J. L. Ba: "ADAM: A Method for Stochastic Optimization."
     * https://arxiv.org/abs/1412.6980v8
     *
     * - "Fixing Weight Decay Regularization in Adam" https://arxiv.org/pdf/1711.05101.pdf
     *
     * - "On the Convergence of Adam and Beyond" https://openreview.net/forum?id=ryQu7f-RZ
     */
    static abstract class Adam extends OptimizerBase {

        protected float alpha;
        protected final float beta1, beta2;
        protected final float eps;
        protected final float decay;

        protected final boolean amsgrad;
        protected float max_vhat = Float.MIN_VALUE;

        public Adam(@Nonnull Map<String, String> options) {
            super(options);
            //this.alpha = Primitives.parseFloat(options.get("alpha"), 0.001f);
            this.alpha = Primitives.parseFloat(options.get("alpha"), 1.0f);
            this.beta1 = Primitives.parseFloat(options.get("beta1"), 0.9f);
            this.beta2 = Primitives.parseFloat(options.get("beta2"), 0.999f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1e-8f);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.f);
            this.amsgrad = options.containsKey("amsgrad");
        }

        @Override
        protected float eta(final long t) {
            double fix1 = 1.d - Math.pow(beta1, t);
            double fix2 = 1.d - Math.pow(beta2, t);
            float eta = _eta.eta(t);
            double fix = Math.sqrt(fix2) / fix1;
            return (float) (eta * fix);
        }

        protected float alpha() {
            double fix1 = 1.d - Math.pow(beta1, _numStep);
            double fix2 = 1.d - Math.pow(beta2, _numStep);
            double fix = Math.sqrt(fix2) / fix1;
            return (float) (alpha * fix);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, float gradient) {
            if (decay != 0.f) {// L2 regularization for weight decay
                float oldWeight = weight.get();
                gradient += decay * oldWeight;
            }
            // update biased first moment estimate
            float m = beta1 * weight.getM() + (1.f - beta1) * gradient;
            // update biased second raw moment estimate
            float v = beta2 * weight.getV() + (float) ((1.f - beta2) * MathUtils.square(gradient));
            float v_hat = v;
            if (amsgrad) {
                if (v_hat > max_vhat) {
                    this.max_vhat = v_hat;
                } else {// v_hat <= max_vhat
                    v_hat = max_vhat;
                }
            }
            // bias correlation using v_hat and m_hat
            float deltaU = m / (float) (Math.sqrt(v_hat) + eps);
            // compute delta update
            float alpha_t = alpha();
            float delta = alpha_t * deltaU;
            // weight decay
            if (decay != 0.f) {
                float oldWeight = weight.get();
                delta += decay * oldWeight;
            }
            weight.setM(m);
            weight.setV(v);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "adam";
        }

    }

    /**
     * Adam optimizer with Hypergradient Descent.
     *
     * - Online Learning Rate Adaptation with Hypergradient Descent
     * https://openreview.net/forum?id=BkrsAzWAb
     *
     * - Convergence Analysis of an Adaptive Method of Gradient Descent
     * https://damaru2.github.io/convergence_analysis_hypergradient_descent/dissertation_hypergradients.pdf
     */
    static abstract class AdamHD extends Adam {

        private final float beta;
        protected float deltaU = 0.f;

        public AdamHD(@Nonnull Map<String, String> options) {
            super(options);
            this.alpha = Primitives.parseFloat(options.get("alpha"), 0.01f);
            this.beta = Primitives.parseFloat(options.get("beta"), 0.0001f);
        }

        private float alpha(final float gradient, final float deltaU) {
            // multiplicative hypergradient descent
            final float h = gradient * deltaU;
            if (h > 0) {// g_{t-1}u_{t-2} > 0
                this.alpha = alpha * (1.f - beta); // decrease alpha
            } else if (h < 0) {// g_{t-1}u_{t-2} < 0
                this.alpha = alpha * (1.f + beta); // increase alpha
            }
            return alpha;
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, float gradient) {
            if (decay != 0.f) {// L2 regularization for weight decay
                float oldWeight = weight.get();
                gradient += decay * oldWeight;
            }
            // update biased first moment estimate
            float m = beta1 * weight.getM() + (1.f - beta1) * gradient;
            // update biased second raw moment estimate
            float v = beta2 * weight.getV() + (float) ((1.f - beta2) * MathUtils.square(gradient));
            float v_hat = v;
            if (amsgrad) {
                if (v_hat > max_vhat) {
                    this.max_vhat = v_hat;
                } else {// v_hat <= max_vhat
                    v_hat = max_vhat;
                }
            }
            // bias correlation using m_hat and v_hat
            float deltaU = m / (float) (Math.sqrt(v_hat) + eps);
            // compute delta update
            float alpha_t = alpha(gradient, deltaU);
            float delta = alpha_t * deltaU;
            this.deltaU = deltaU;
            // weight decay
            if (decay != 0.f) {
                float oldWeight = weight.get();
                delta += decay * oldWeight;
            }
            weight.setM(m);
            weight.setV(v);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "adam-hd";
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
