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

import static hivemall.utils.math.MathUtils.square;
import static java.lang.Math.abs;
import static java.lang.Math.floor;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import hivemall.model.IWeightValue;
import hivemall.model.WeightValue;
import hivemall.model.WeightValue.WeightValueParamsF1;
import hivemall.model.WeightValue.WeightValueParamsF2;
import hivemall.model.WeightValue.WeightValueParamsF3;
import hivemall.utils.lang.Primitives;
import hivemall.utils.math.MathUtils;

import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

public interface Optimizer {

    /**
     * Update the weights of models
     */
    float update(@Nonnull Object feature, float weight, float loss, float gradient);

    /**
     * Count up #step to tune learning rate
     */
    void proceedStep();

    @Nonnull
    String getOptimizerName();

    @Nonnull
    Map<String, Object> getHyperParameters();

    @NotThreadSafe
    static abstract class OptimizerBase implements Optimizer {

        @Nonnull
        protected final EtaEstimator _eta;
        @Nonnull
        protected final Regularization _reg;
        @Nonnegative
        protected long _numStep = 0L;

        public OptimizerBase(@Nonnull Map<String, String> options) {
            this._eta = getEtaEstimator(options);
            this._reg = Regularization.get(options);
        }

        @Nonnull
        protected abstract IWeightValue newWeightValue(final float weight);

        @Nonnull
        protected EtaEstimator getEtaEstimator(@Nonnull Map<String, String> options) {
            return EtaEstimator.get(options);
        }

        @Override
        public void proceedStep() {
            _numStep++;
        }

        @Override
        public float update(@Nonnull Object feature, float weight, float loss, float gradient) {
            return update(feature, weight, gradient);
        }

        /**
         * Update the weights of models
         */
        protected abstract float update(@Nonnull Object feature, float weight, float gradient);

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
         * @param t time step
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

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = new HashMap<>();
            params.put("optimizer", getOptimizerName());
            _eta.getHyperParameters(params);
            _reg.getHyperParameters(params);
            return params;
        }

    }

    static final class SGD extends OptimizerBase {

        private final IWeightValue weightValueReused;

        public SGD(@Nonnull Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
        }

        @Override
        protected WeightValue newWeightValue(final float weight) {
            return new WeightValue(weight);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
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

    /**
     * Momentum and Nesterov's Accelerated Gradient.
     *
     * https://arxiv.org/abs/1212.0901
     */
    static abstract class Momentum extends OptimizerBase {

        @Nonnull
        private final WeightValueParamsF1 weightValueReused;

        private final boolean nesterov;
        private final float alpha;
        private final float momentum;

        public Momentum(@Nonnull Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.nesterov = options.containsKey("nesterov");
            this.alpha = Primitives.parseFloat(options.get("alpha"), 1.f);
            this.momentum = Primitives.parseFloat(options.get("momentum"), 0.9f);
        }

        @Override
        protected WeightValueParamsF1 newWeightValue(final float weight) {
            return new WeightValueParamsF1(weight, 0.f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            final float oldDelta = weight.getDelta();
            final float v = momentum * oldDelta + alpha * gradient;
            weight.setDelta(v);
            if (nesterov) {
                //return momentum * momentum * oldDelta + (1.f + momentum) * alpha * gradient;
                return momentum * momentum * v + (1.f + momentum) * alpha * gradient;
            } else {
                return v; // normal momentum
            }
        }

        @Override
        public String getOptimizerName() {
            return nesterov ? "nesterov" : "momentum";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("nesterov", nesterov);
            params.put("alpha", alpha);
            params.put("momentum", momentum);
            return params;
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
        protected WeightValueParamsF1 newWeightValue(final float weight) {
            return new WeightValueParamsF1(weight, 0.f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_gg = weight.getSumOfSquaredGradients();
            float new_scaled_gg = old_scaled_gg + gradient * (gradient / scale);
            weight.setSumOfSquaredGradients(new_scaled_gg);
            return (float) (gradient / sqrt(eps + ((double) old_scaled_gg) * scale));
        }

        @Override
        public String getOptimizerName() {
            return "adagrad";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("eps", eps);
            params.put("scale", scale);
            return params;
        }
    }

    /**
     * RMSprop optimizer introducing weight decay to AdaGrad.
     *
     * Geoffrey Hinton, Nitish Srivastava, Kevin Swersky. 2014. "Lecture 6e: Rmsprop: Divide the
     * gradient by a running average of its recent magnitude"
     *
     * @see http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
     */
    static abstract class RMSprop extends OptimizerBase {

        /** decay rate */
        private final float decay;
        /** constant for numerical stability */
        private final float eps;

        private final float scale; // to hold g*g in float range

        public RMSprop(@Nonnull Map<String, String> options) {
            super(options);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.95f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1.0f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected WeightValueParamsF1 newWeightValue(final float weight) {
            return new WeightValueParamsF1(weight, 0.f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_gg = weight.getSumOfSquaredGradients();
            float new_scaled_gg =
                    decay * old_scaled_gg + (1.f - decay) * gradient * (gradient / scale);
            weight.setSumOfSquaredGradients(new_scaled_gg);
            return (float) (gradient / sqrt(eps + ((double) old_scaled_gg) * scale));
        }

        @Override
        public String getOptimizerName() {
            return "rmsprop";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("decay", decay);
            params.put("eps", eps);
            params.put("scale", scale);
            return params;
        }

    }

    /**
     * Alex Graves's RMSprop introducing weight decay and momentum.
     *
     * @see https://arxiv.org/abs/1308.0850
     */
    static abstract class RMSpropGraves extends OptimizerBase {

        /** decay rate */
        private final float decay;
        private final float alpha;
        private final float momentum;
        /** constant for numerical stability */
        private final float eps;

        private final float scale; // to hold g*g in float range

        public RMSpropGraves(@Nonnull Map<String, String> options) {
            super(options);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.95f);
            this.alpha = Primitives.parseFloat(options.get("alpha"), 1.f);
            this.momentum = Primitives.parseFloat(options.get("momentum"), 0.9f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1.0f);
            this.scale = Primitives.parseFloat(options.get("scale"), 100.0f);
        }

        @Override
        protected WeightValueParamsF3 newWeightValue(final float weight) {
            return new WeightValueParamsF3(weight, 0.f, 0.f, 0.f);
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, final float gradient) {
            float old_scaled_n = weight.getSumOfSquaredGradients();
            float new_scaled_n =
                    decay * old_scaled_n + (1.f - decay) * gradient * (gradient / scale);
            weight.setSumOfSquaredGradients(new_scaled_n);
            float old_scaled_g = weight.getSumOfGradients();
            float new_scaled_g = decay * old_scaled_g + (1.f - decay) * gradient / scale;
            weight.setSumOfGradients(new_scaled_g);
            double n = ((double) old_scaled_n) * scale;
            double g = ((double) new_scaled_g) * scale;
            float oldDelta = weight.getDelta();
            float delta = momentum * oldDelta + alpha * (float) (gradient / sqrt(n - g * g + eps));
            weight.setDelta(delta);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "rmsprop_graves";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("decay", decay);
            params.put("alpha", alpha);
            params.put("momentum", momentum);
            params.put("eps", eps);
            return params;
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
        protected WeightValueParamsF2 newWeightValue(final float weight) {
            return new WeightValueParamsF2(weight, 0.f, 0.f);
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
            float delta = (float) sqrt(
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

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("decay", decay);
            params.put("eps", eps);
            params.put("scale", scale);
            return params;
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

        // amsgrad
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
        protected WeightValueParamsF2 newWeightValue(final float weight) {
            return new WeightValueParamsF2(weight, 0.f, 0.f);
        }

        @Override
        protected float eta(final long t) {
            double fix1 = 1.d - pow(beta1, t);
            double fix2 = 1.d - pow(beta2, t);
            float eta = _eta.eta(t);
            double fix = sqrt(fix2) / fix1;
            return (float) (eta * fix);
        }

        protected double alpha() {
            double fix1 = 1.d - pow(beta1, _numStep);
            double fix2 = 1.d - pow(beta2, _numStep);
            double fix = sqrt(fix2) / fix1;
            return alpha * fix;
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
            float v = beta2 * weight.getV() + (float) ((1.f - beta2) * square(gradient));
            float v_hat = v;
            if (amsgrad) {
                if (v_hat > max_vhat) {
                    this.max_vhat = v_hat;
                } else {// v_hat <= max_vhat
                    v_hat = max_vhat;
                }
            }
            // bias correlation using v_hat and m_hat
            double deltaU = m / (sqrt(v_hat) + eps);
            // compute delta update
            double alpha_t = alpha();
            float delta = (float) (alpha_t * deltaU);
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
            return amsgrad ? "adam-amsgrad" : "adam";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("alpha", alpha);
            params.put("beta1", beta1);
            params.put("beta2", beta2);
            params.put("eps", eps);
            params.put("decay", decay);
            return params;
        }
    }

    /**
     * Nadam is Adam optimizer with Nesterov momentum.
     *
     * @see https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
     * @see http://cs229.stanford.edu/proj2015/054_report.pdf
     * @see http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
     */
    static abstract class Nadam extends OptimizerBase {

        protected float alpha;
        protected final float beta1, beta2;
        protected final float eps;
        protected final float decay;
        protected final float scheduleDecay;

        protected double mu_t, mu_t_1;
        protected double mu_product = 1.d;
        protected double mu_product_next = 1.d;

        public Nadam(@Nonnull Map<String, String> options) {
            super(options);
            //this.alpha = Primitives.parseFloat(options.get("alpha"), 0.001f);
            this.alpha = Primitives.parseFloat(options.get("alpha"), 1.0f);
            this.beta1 = Primitives.parseFloat(options.get("beta1"), 0.9f);
            this.beta2 = Primitives.parseFloat(options.get("beta2"), 0.999f);
            this.eps = Primitives.parseFloat(options.get("eps"), 1e-8f);
            this.decay = Primitives.parseFloat(options.get("decay"), 0.f);
            this.scheduleDecay = Primitives.parseFloat(options.get("scheduleDecay"), 0.004f); // 1/250=0.004
        }

        @Override
        protected WeightValueParamsF2 newWeightValue(final float weight) {
            return new WeightValueParamsF2(weight, 0.f, 0.f);
        }

        @Override
        public void proceedStep() {
            long t = _numStep + 1;
            this._numStep = t;
            double mu_product_prev = this.mu_product;
            // 0.9 * (1 - 0.5 * 0.96^(floor(t/250)+1))
            double mu_t = beta1 * (1.d - 0.5d * pow(0.96d, floor(t * scheduleDecay) + 1.d));
            double mu_t_1 =
                    beta1 * (1.d - 0.5d * pow(0.96d, floor((t + 1.d) * scheduleDecay) + 1.d));
            this.mu_t = mu_t;
            this.mu_t_1 = mu_t_1;
            this.mu_product = mu_product_prev * mu_t;
            this.mu_product_next = mu_product_prev * mu_t * mu_t_1;
        }

        @Override
        protected float eta(final long t) {
            double fix1 = 1.d - pow(beta1, t);
            double fix2 = 1.d - pow(beta2, t);
            float eta = _eta.eta(t);
            double fix = sqrt(fix2) / fix1;
            return (float) (eta * fix);
        }

        protected double alpha() {
            double fix1 = 1.d - pow(beta1, _numStep);
            double fix2 = 1.d - pow(beta2, _numStep);
            double fix = sqrt(fix2) / fix1;
            return alpha * fix;
        }

        @Override
        protected float computeDelta(@Nonnull final IWeightValue weight, float gradient) {
            if (decay != 0.f) {// L2 regularization for weight decay
                float oldWeight = weight.get();
                gradient += decay * oldWeight;
            }
            // update biased first moment estimate
            float m = beta1 * weight.getM() + (1.f - beta1) * gradient;
            double m_hat = m / (1.d - mu_product_next);
            // update biased second raw moment estimate
            float v = beta2 * weight.getV() + (float) ((1.d - beta2) * square(gradient));
            double v_hat = v / (1.d - pow(beta2, _numStep));
            // gradient update for the current timestamp
            double g_hat = gradient / (1.d - mu_product);
            double m_bar = (1.d - mu_t) * g_hat + mu_t_1 * m_hat;
            // bias correlation using v_hat and m_hat
            double deltaU = m_bar / (sqrt(v_hat) + eps);
            // compute delta update
            double alpha_t = alpha();
            float delta = (float) (alpha_t * deltaU);
            // weight decay
            if (decay != 0.d) {
                float oldWeight = weight.get();
                delta += decay * oldWeight;
            }
            weight.setM(m);
            weight.setV(v);
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "nadam";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("alpha", alpha);
            params.put("beta1", beta1);
            params.put("beta2", beta2);
            params.put("eps", eps);
            params.put("decay", decay);
            params.put("scheduleDecay", scheduleDecay);
            return params;
        }
    }

    /**
     * Eve optimizer.
     *
     * - "Eve: A Gradient Based Optimization Method with Locally and Globally Adaptive Learning
     * Rates" https://openreview.net/forum?id=r1WUqIceg
     */
    static abstract class Eve extends Adam {

        protected final float beta3;
        private float c = 10.f;
        private float inv_c = 0.1f;

        private float currLoss;
        private float prevLoss = 0.f;
        private double prevDt = 1.d;

        public Eve(@Nonnull Map<String, String> options) {
            super(options);
            this.beta3 = Primitives.parseFloat(options.get("beta3"), 0.999f);
            this.c = Primitives.parseFloat(options.get("c"), 10f);
            this.inv_c = 1f / c;
        }

        @Override
        protected double alpha() {
            double fix1 = 1.d - pow(beta1, _numStep);
            double fix2 = 1.d - pow(beta2, _numStep);
            double fix = sqrt(fix2) / fix1;
            double alpha_t = alpha * fix;
            // feedback of Eve
            if (_numStep > 1 && currLoss != prevLoss) {
                double d = abs(currLoss - prevLoss) / min(currLoss, prevLoss);
                d = MathUtils.clip(d, inv_c, c); // [alpha/c, c*alpha]
                d = (beta3 * prevDt) + (1.d - beta3) * d;
                this.prevDt = d;
                alpha_t = alpha_t / d;
            }
            return alpha_t;
        }

        @Override
        public float update(Object feature, float weight, float loss, float gradient) {
            this.currLoss = loss;
            float delta = update(feature, weight, gradient);
            this.prevLoss = loss;
            return delta;
        }

        @Override
        public String getOptimizerName() {
            return "eve";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("beta3", beta3);
            params.put("c", c);
            return params;
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
        protected double deltaU = 0.d;

        public AdamHD(@Nonnull Map<String, String> options) {
            super(options);
            this.alpha = Primitives.parseFloat(options.get("alpha"), 0.02f);
            this.beta = Primitives.parseFloat(options.get("beta"), 1e-6f);
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

        private float alpha(final float gradient, final double deltaU) {
            // multiplicative hypergradient descent
            final double h = gradient * deltaU;
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
            float v = beta2 * weight.getV() + (float) ((1.f - beta2) * square(gradient));
            // compute bias-corrected first moment estimate
            double m_hat = m / (1.d - pow(beta1, _numStep));
            // compute bias-corrected second raw moment estimate
            double v_hat = v / (1.d - pow(beta2, _numStep));
            // compute delta update
            float alpha_t = alpha(gradient, deltaU);
            double deltaU = m_hat / (sqrt(v_hat) + eps);
            float delta = (float) (alpha_t * deltaU);
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
            return "adam_hd";
        }

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = super.getHyperParameters();
            params.put("beta", beta);
            return params;
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
        protected WeightValueParamsF2 newWeightValue(final float weight) {
            return new WeightValueParamsF2(weight, 0.f, 0.f);
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

        @Override
        public Map<String, Object> getHyperParameters() {
            Map<String, Object> params = optimizerImpl.getHyperParameters();
            params.put("optimizer", getOptimizerName()); // replace
            params.put("lambda", lambda);
            return params;
        }
    }

}
