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
import hivemall.optimizer.Optimizer.OptimizerBase;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;

import java.util.Map;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class SparseOptimizerFactory {
    private static final Log LOG = LogFactory.getLog(SparseOptimizerFactory.class);

    @Nonnull
    public static Optimizer create(@Nonnull final int ndims,
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
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public Momentum(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class AdaGrad extends Optimizer.AdaGrad {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public AdaGrad(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class RMSprop extends Optimizer.RMSprop {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public RMSprop(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class RMSpropGraves extends Optimizer.RMSpropGraves {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public RMSpropGraves(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class AdaDelta extends Optimizer.AdaDelta {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public AdaDelta(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class Adam extends Optimizer.Adam {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public Adam(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class Nadam extends Optimizer.Nadam {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public Nadam(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class Eve extends Optimizer.Eve {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public Eve(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class AdamHD extends Optimizer.AdamHD {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public AdamHD(@Nonnegative int size, @Nonnull Map<String, String> options) {
            super(options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

    @NotThreadSafe
    static final class AdagradRDA extends Optimizer.AdagradRDA {

        @Nonnull
        private final Object2ObjectMap<Object, IWeightValue> auxWeights;

        public AdagradRDA(@Nonnegative int size, @Nonnull Optimizer.AdaGrad optimizerImpl,
                @Nonnull Map<String, String> options) {
            super(optimizerImpl, options);
            this.auxWeights = new Object2ObjectOpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        protected float update(@Nonnull final Object feature, final float weight,
                final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = newWeightValue(weight);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            final float newWeight = update(auxWeight, gradient);
            if (newWeight == 0.f) {
                auxWeights.remove(feature);
            }
            return newWeight;
        }

    }

}
