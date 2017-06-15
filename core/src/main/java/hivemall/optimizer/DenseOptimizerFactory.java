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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class DenseOptimizerFactory {
    private static final Log LOG = LogFactory.getLog(DenseOptimizerFactory.class);

    @Nonnull
    public static Optimizer create(int ndims, @Nonnull Map<String, String> options) {
        final String optimizerName = options.get("optimizer");
        if (optimizerName == null) {
            throw new IllegalArgumentException("`optimizer` not defined");
        }

        if ("rda".equalsIgnoreCase(options.get("regularization"))
                && "adagrad".equalsIgnoreCase(optimizerName) == false) {
            throw new IllegalArgumentException(
                "`-regularization rda` is only supported for AdaGrad but `-optimizer "
                        + optimizerName);
        }

        final Optimizer optimizerImpl;
        if ("sgd".equalsIgnoreCase(optimizerName)) {
            optimizerImpl = new Optimizer.SGD(options);
        } else if ("adadelta".equalsIgnoreCase(optimizerName)) {
            optimizerImpl = new AdaDelta(ndims, options);
        } else if ("adagrad".equalsIgnoreCase(optimizerName)) {
            // If a regularization type is "RDA", wrap the optimizer with `Optimizer#RDA`.
            if ("rda".equalsIgnoreCase(options.get("regularization"))) {
                AdaGrad adagrad = new AdaGrad(ndims, options);
                optimizerImpl = new AdagradRDA(ndims, adagrad, options);
            } else {
                optimizerImpl = new AdaGrad(ndims, options);
            }
        } else if ("adam".equalsIgnoreCase(optimizerName)) {
            optimizerImpl = new Adam(ndims, options);
        } else {
            throw new IllegalArgumentException("Unsupported optimizer name: " + optimizerName);
        }

        if (LOG.isInfoEnabled()) {
            LOG.info("Configured " + optimizerImpl.getOptimizerName() + " as the optimizer: "
                    + options);
        }

        return optimizerImpl;
    }

    @NotThreadSafe
    static final class AdaDelta extends Optimizer.AdaDelta {

        @Nonnull
        private final IWeightValue weightValueReused;

        @Nonnull
        private float[] sum_of_squared_gradients;
        @Nonnull
        private float[] sum_of_squared_delta_x;

        public AdaDelta(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = new WeightValue.WeightValueParamsF2(0.f, 0.f, 0.f);
            this.sum_of_squared_gradients = new float[ndims];
            this.sum_of_squared_delta_x = new float[ndims];
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
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
    static final class AdaGrad extends Optimizer.AdaGrad {

        @Nonnull
        private final IWeightValue weightValueReused;
        @Nonnull
        private float[] sum_of_squared_gradients;

        public AdaGrad(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = new WeightValue.WeightValueParamsF1(0.f, 0.f);
            this.sum_of_squared_gradients = new float[ndims];
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
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
    static final class Adam extends Optimizer.Adam {

        @Nonnull
        private final IWeightValue weightValueReused;

        @Nonnull
        private float[] val_m;
        @Nonnull
        private float[] val_v;

        public Adam(int ndims, Map<String, String> options) {
            super(options);
            this.weightValueReused = new WeightValue.WeightValueParamsF2(0.f, 0.f, 0.f);
            this.val_m = new float[ndims];
            this.val_v = new float[ndims];
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
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
        private final IWeightValue weightValueReused;

        @Nonnull
        private float[] sum_of_gradients;

        public AdagradRDA(int ndims, @Nonnull Optimizer.AdaGrad optimizerImpl,
                @Nonnull Map<String, String> options) {
            super(optimizerImpl, options);
            this.weightValueReused = new WeightValue.WeightValueParamsF3(0.f, 0.f, 0.f, 0.f);
            this.sum_of_gradients = new float[ndims];
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
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
