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

import hivemall.model.WeightValue.WeightValueParamsF1;
import hivemall.model.WeightValue.WeightValueParamsF2;
import hivemall.model.WeightValue.WeightValueParamsF3;
import hivemall.optimizer.Optimizer.OptimizerBase;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;
import java.util.Map;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class DenseOptimizerFactory {
    private static final Log LOG = LogFactory.getLog(DenseOptimizerFactory.class);

    @Nonnull
    public static Optimizer create(@Nonnegative final int ndims,
            @Nonnull final Map<String, String> options) {
        final String optimizerName = options.get("optimizer");
        if (optimizerName == null) {
            throw new IllegalArgumentException("`optimizer` not defined");
        }
        final String name = optimizerName.toLowerCase();

        if ("rda".equalsIgnoreCase(options.get("regularization"))
                && "adagrad".equals(name) == false) {
            throw new IllegalArgumentException(
                "`-regularization rda` is only supported for AdaGrad but `-optimizer "
                        + optimizerName + "`. Please specify `-regularization l1` and so on.");
        }

        final OptimizerBase optimizerImpl;
        if ("sgd".equals(name)) {
            optimizerImpl = new Optimizer.SGD(options);
        } else if ("momentum".equals(name)) {
            optimizerImpl = new Momentum(ndims, options);
        } else if ("nesterov".equals(name)) {
            options.put("nesterov", "");
            optimizerImpl = new Momentum(ndims, options);
        } else if ("adagrad".equals(name)) {
            // If a regularization type is "RDA", wrap the optimizer with `Optimizer#RDA`.
            if ("rda".equalsIgnoreCase(options.get("regularization"))) {
                AdaGrad adagrad = new AdaGrad(ndims, options);
                optimizerImpl = new AdagradRDA(ndims, adagrad, options);
            } else {
                optimizerImpl = new AdaGrad(ndims, options);
            }
        } else if ("rmsprop".equals(name)) {
            optimizerImpl = new RMSprop(ndims, options);
        } else if ("rmspropgraves".equals(name) || "rmsprop_graves".equals(name)) {
            optimizerImpl = new RMSpropGraves(ndims, options);
        } else if ("adadelta".equals(name)) {
            optimizerImpl = new AdaDelta(ndims, options);
        } else if ("adam".equals(name)) {
            optimizerImpl = new Adam(ndims, options);
        } else if ("nadam".equals(name)) {
            optimizerImpl = new Nadam(ndims, options);
        } else if ("eve".equals(name)) {
            optimizerImpl = new Eve(ndims, options);
        } else if ("adam_hd".equals(name) || "adamhd".equals(name)) {
            optimizerImpl = new AdamHD(ndims, options);
        } else {
            throw new IllegalArgumentException("Unsupported optimizer name: " + optimizerName);
        }

        if (LOG.isInfoEnabled()) {
            LOG.info(
                "Configured " + optimizerImpl.getOptimizerName() + " as the optimizer: " + options);
            LOG.info("ETA estimator: " + optimizerImpl._eta);
        }

        return optimizerImpl;
    }

    @NotThreadSafe
    static final class Momentum extends Optimizer.Momentum {

        @Nonnull
        private final WeightValueParamsF1 weightValueReused;
        @Nonnull
        private float[] delta;

        public Momentum(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.delta = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setDelta(delta[i]);
            update(weightValueReused, gradient);
            delta[i] = weightValueReused.getDelta();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= delta.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.delta = Arrays.copyOf(delta, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class AdaGrad extends Optimizer.AdaGrad {

        @Nonnull
        private final WeightValueParamsF1 weightValueReused;
        @Nonnull
        private float[] sum_of_squared_gradients;

        public AdaGrad(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.sum_of_squared_gradients = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setSumOfSquaredGradients(sum_of_squared_gradients[i]);
            update(weightValueReused, gradient);
            sum_of_squared_gradients[i] = weightValueReused.getSumOfSquaredGradients();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= sum_of_squared_gradients.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.sum_of_squared_gradients = Arrays.copyOf(sum_of_squared_gradients, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class RMSprop extends Optimizer.RMSprop {

        @Nonnull
        private final WeightValueParamsF1 weightValueReused;
        @Nonnull
        private float[] sum_of_squared_gradients;

        public RMSprop(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.sum_of_squared_gradients = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setSumOfSquaredGradients(sum_of_squared_gradients[i]);
            update(weightValueReused, gradient);
            sum_of_squared_gradients[i] = weightValueReused.getSumOfSquaredGradients();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= sum_of_squared_gradients.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.sum_of_squared_gradients = Arrays.copyOf(sum_of_squared_gradients, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class RMSpropGraves extends Optimizer.RMSpropGraves {

        @Nonnull
        private final WeightValueParamsF3 weightValueReused;
        @Nonnull
        private float[] sum_of_gradients;
        @Nonnull
        private float[] sum_of_squared_gradients;
        @Nonnull
        private float[] delta;

        public RMSpropGraves(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.sum_of_gradients = new float[ndims];
            this.sum_of_squared_gradients = new float[ndims];
            this.delta = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setSumOfGradients(sum_of_gradients[i]);
            weightValueReused.setSumOfSquaredGradients(sum_of_squared_gradients[i]);
            weightValueReused.setDelta(delta[i]);
            update(weightValueReused, gradient);
            sum_of_gradients[i] = weightValueReused.getSumOfGradients();
            sum_of_squared_gradients[i] = weightValueReused.getSumOfSquaredGradients();
            delta[i] = weightValueReused.getDelta();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= sum_of_gradients.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.sum_of_gradients = Arrays.copyOf(sum_of_gradients, newSize);
                this.sum_of_squared_gradients = Arrays.copyOf(sum_of_squared_gradients, newSize);
                this.delta = Arrays.copyOf(delta, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class AdaDelta extends Optimizer.AdaDelta {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] sum_of_squared_gradients;
        @Nonnull
        private float[] sum_of_squared_delta_x;

        public AdaDelta(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.sum_of_squared_gradients = new float[ndims];
            this.sum_of_squared_delta_x = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setSumOfSquaredGradients(sum_of_squared_gradients[i]);
            weightValueReused.setSumOfSquaredDeltaX(sum_of_squared_delta_x[i]);
            update(weightValueReused, gradient);
            sum_of_squared_gradients[i] = weightValueReused.getSumOfSquaredGradients();
            sum_of_squared_delta_x[i] = weightValueReused.getSumOfSquaredDeltaX();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= sum_of_squared_gradients.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.sum_of_squared_gradients = Arrays.copyOf(sum_of_squared_gradients, newSize);
                this.sum_of_squared_delta_x = Arrays.copyOf(sum_of_squared_delta_x, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class Adam extends Optimizer.Adam {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] val_m;
        @Nonnull
        private float[] val_v;

        public Adam(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.val_m = new float[ndims];
            this.val_v = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setM(val_m[i]);
            weightValueReused.setV(val_v[i]);
            update(weightValueReused, gradient);
            val_m[i] = weightValueReused.getM();
            val_v[i] = weightValueReused.getV();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= val_m.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.val_m = Arrays.copyOf(val_m, newSize);
                this.val_v = Arrays.copyOf(val_v, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class Nadam extends Optimizer.Nadam {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] val_m;
        @Nonnull
        private float[] val_v;

        public Nadam(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.val_m = new float[ndims];
            this.val_v = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setM(val_m[i]);
            weightValueReused.setV(val_v[i]);
            update(weightValueReused, gradient);
            val_m[i] = weightValueReused.getM();
            val_v[i] = weightValueReused.getV();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= val_m.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.val_m = Arrays.copyOf(val_m, newSize);
                this.val_v = Arrays.copyOf(val_v, newSize);
            }
        }

    }

    @NotThreadSafe
    static final class Eve extends Optimizer.Eve {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] val_m;
        @Nonnull
        private float[] val_v;

        public Eve(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.val_m = new float[ndims];
            this.val_v = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setM(val_m[i]);
            weightValueReused.setV(val_v[i]);
            update(weightValueReused, gradient);
            val_m[i] = weightValueReused.getM();
            val_v[i] = weightValueReused.getV();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= val_m.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.val_m = Arrays.copyOf(val_m, newSize);
                this.val_v = Arrays.copyOf(val_v, newSize);
            }
        }

    }


    @NotThreadSafe
    static final class AdamHD extends Optimizer.AdamHD {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] val_m;
        @Nonnull
        private float[] val_v;

        public AdamHD(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = newWeightValue(0.f);
            this.val_m = new float[ndims];
            this.val_v = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setM(val_m[i]);
            weightValueReused.setV(val_v[i]);
            update(weightValueReused, gradient);
            val_m[i] = weightValueReused.getM();
            val_v[i] = weightValueReused.getV();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= val_m.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.val_m = Arrays.copyOf(val_m, newSize);
                this.val_v = Arrays.copyOf(val_v, newSize);
            }
        }
    }

    @NotThreadSafe
    static final class AdagradRDA extends Optimizer.AdagradRDA {

        @Nonnull
        private final WeightValueParamsF2 weightValueReused;

        @Nonnull
        private float[] sum_of_gradients;

        public AdagradRDA(int ndims, @Nonnull Optimizer.AdaGrad optimizerImpl,
                @Nonnull Map<String, String> options) {
            super(optimizerImpl, options);
            this.weightValueReused = newWeightValue(0.f);
            this.sum_of_gradients = new float[ndims];
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            int i = HiveUtils.parseInt(feature);
            ensureCapacity(i);
            weightValueReused.set(weight);
            weightValueReused.setSumOfGradients(sum_of_gradients[i]);
            update(weightValueReused, gradient);
            sum_of_gradients[i] = weightValueReused.getSumOfGradients();
            return weightValueReused.get();
        }

        private void ensureCapacity(final int index) {
            if (index >= sum_of_gradients.length) {
                int bits = MathUtils.bitsRequired(index);
                int newSize = (1 << bits) + 1;
                this.sum_of_gradients = Arrays.copyOf(sum_of_gradients, newSize);
            }
        }

    }

}
