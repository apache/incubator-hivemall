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
import hivemall.optimizer.Optimizer.OptimizerBase;
import hivemall.utils.collections.OpenHashMap;

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
        if (optimizerName != null) {
            OptimizerBase optimizerImpl;
            if (optimizerName.toLowerCase().equals("sgd")) {
                optimizerImpl = new Optimizer.SGD(options);
            } else if (optimizerName.toLowerCase().equals("adadelta")) {
                optimizerImpl = new AdaDelta(ndims, options);
            } else if (optimizerName.toLowerCase().equals("adagrad")) {
                optimizerImpl = new AdaGrad(ndims, options);
            } else if (optimizerName.toLowerCase().equals("adam")) {
                optimizerImpl = new Adam(ndims, options);
            } else {
                throw new IllegalArgumentException("Unsupported optimizer name: " + optimizerName);
            }

            // If a regularization type is "RDA", wrap the optimizer with `Optimizer#RDA`.
            if (options.get("regularization") != null
                    && options.get("regularization").toLowerCase().equals("rda")) {
                optimizerImpl = new AdagradRDA(ndims, optimizerImpl, options);
            }

            if (LOG.isInfoEnabled()) {
                LOG.info("set " + optimizerImpl.getClass().getSimpleName() + " as an optimizer: "
                        + options);
            }

            return optimizerImpl;
        }
        throw new IllegalArgumentException("`optimizer` not defined");
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
        public float update(@Nonnull Object feature, float weight, float gradient) {
            IWeightValue auxWeight;
            if (auxWeights.containsKey(feature)) {
                auxWeight = auxWeights.get(feature);
                auxWeight.set(weight);
            } else {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
                auxWeights.put(feature, auxWeight);
            }
            update(auxWeight, gradient);
            return auxWeight.get();
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
        public float update(@Nonnull Object feature, float weight, float gradient) {
            IWeightValue auxWeight;
            if (auxWeights.containsKey(feature)) {
                auxWeight = auxWeights.get(feature);
                auxWeight.set(weight);
            } else {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
                auxWeights.put(feature, auxWeight);
            }
            update(auxWeight, gradient);
            return auxWeight.get();
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
        public float update(@Nonnull Object feature, float weight, float gradient) {
            IWeightValue auxWeight;
            if (auxWeights.containsKey(feature)) {
                auxWeight = auxWeights.get(feature);
                auxWeight.set(weight);
            } else {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
                auxWeights.put(feature, auxWeight);
            }
            update(auxWeight, gradient);
            return auxWeight.get();
        }

    }

    @NotThreadSafe
    static final class AdagradRDA extends Optimizer.AdagradRDA {

        @Nonnull
        private final OpenHashMap<Object, IWeightValue> auxWeights;

        public AdagradRDA(int size, OptimizerBase optimizerImpl, Map<String, String> options) {
            super(optimizerImpl, options);
            this.auxWeights = new OpenHashMap<Object, IWeightValue>(size);
        }

        @Override
        public float update(@Nonnull Object feature, float weight, float gradient) {
            IWeightValue auxWeight;
            if (auxWeights.containsKey(feature)) {
                auxWeight = auxWeights.get(feature);
                auxWeight.set(weight);
            } else {
                auxWeight = new WeightValue.WeightValueParamsF2(weight, 0.f, 0.f);
                auxWeights.put(feature, auxWeight);
            }
            update(auxWeight, gradient);
            return auxWeight.get();
        }

    }

}
