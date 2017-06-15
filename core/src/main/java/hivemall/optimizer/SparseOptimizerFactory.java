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
import hivemall.utils.collections.maps.OpenHashMap;

import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class SparseOptimizerFactory {
    private static final Log LOG = LogFactory.getLog(SparseOptimizerFactory.class);

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
        private final OpenHashMap<Object, IWeightValue> auxWeights;

        public AdaDelta(int size, Map<String, String> options) {
            super(options);
            this.auxWeights = new OpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
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
        private final OpenHashMap<Object, IWeightValue> auxWeights;

        public AdaGrad(int size, Map<String, String> options) {
            super(options);
            this.auxWeights = new OpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
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
        private final OpenHashMap<Object, IWeightValue> auxWeights;

        public Adam(int size, Map<String, String> options) {
            super(options);
            this.auxWeights = new OpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
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
        private final OpenHashMap<Object, IWeightValue> auxWeights;

        public AdagradRDA(int size, @Nonnull Optimizer.AdaGrad optimizerImpl,
                @Nonnull Map<String, String> options) {
            super(optimizerImpl, options);
            this.auxWeights = new OpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        public float update(@Nonnull final Object feature, final float weight, final float gradient) {
            IWeightValue auxWeight = auxWeights.get(feature);
            if (auxWeight == null) {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
                auxWeights.put(feature, auxWeight);
            } else {
                auxWeight.set(weight);
            }
            return update(auxWeight, gradient);
        }

    }

}
