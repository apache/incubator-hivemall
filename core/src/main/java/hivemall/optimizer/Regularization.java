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

import hivemall.utils.lang.Primitives;

import java.util.Map;

import javax.annotation.Nonnull;

public abstract class Regularization {
    /** the default regularization term 0.0001 */
    private static final float DEFAULT_LAMBDA = 0.0001f;

    protected final float lambda;

    public Regularization(@Nonnull Map<String, String> options) {
        this.lambda = Primitives.parseFloat(options.get("lambda"), DEFAULT_LAMBDA);
    }

    public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
        hyperParams.put("lambda", lambda);
    }

    public float regularize(final float weight, final float gradient) {
        return gradient + lambda * getRegularizer(weight);
    }

    abstract float getRegularizer(float weight);

    public static final class PassThrough extends Regularization {

        public PassThrough(final Map<String, String> options) {
            super(options);
        }

        @Override
        public float getRegularizer(float weight) {
            return 0.f;
        }

        @Override
        public float regularize(final float weight, final float gradient) {
            return gradient;
        }

        @Override
        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("regularization", "no");
        }
    }

    public static final class L1 extends Regularization {

        public L1(Map<String, String> options) {
            super(options);
        }

        @Override
        public float getRegularizer(final float weight) {
            return weight > 0.f ? 1.f : -1.f;
        }

        @Override
        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("regularization", "L1");
        }
    }

    public static final class L2 extends Regularization {

        public L2(final Map<String, String> options) {
            super(options);
        }

        @Override
        public float getRegularizer(float weight) {
            return weight;
        }

        @Override
        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("regularization", "L2");
        }
    }

    public static final class ElasticNet extends Regularization {
        private static final float DEFAULT_L1_RATIO = 0.5f;

        @Nonnull
        private final L1 l1;
        @Nonnull
        private final L2 l2;

        private final float l1Ratio;

        public ElasticNet(@Nonnull Map<String, String> options) {
            super(options);

            this.l1 = new L1(options);
            this.l2 = new L2(options);

            this.l1Ratio = Primitives.parseFloat(options.get("l1_ratio"), DEFAULT_L1_RATIO);
            if (l1Ratio < 0.f || l1Ratio > 1.f) {
                throw new IllegalArgumentException(
                    "L1 ratio should be in [0.0, 1.0], but got " + l1Ratio);
            }
        }

        @Override
        public float getRegularizer(final float weight) {
            return l1Ratio * l1.getRegularizer(weight)
                    + (1.f - l1Ratio) * l2.getRegularizer(weight);
        }

        @Override
        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("regularization", "ElasticNet");
            hyperParams.put("l1_ratio", l1Ratio);
        }
    }

    @Nonnull
    public static Regularization get(@Nonnull final Map<String, String> options)
            throws IllegalArgumentException {
        final String regName = options.get("regularization");
        if (regName == null) {
            return new PassThrough(options);
        }

        if ("no".equalsIgnoreCase(regName)) {
            return new PassThrough(options);
        } else if ("l1".equalsIgnoreCase(regName)) {
            return new L1(options);
        } else if ("l2".equalsIgnoreCase(regName)) {
            return new L2(options);
        } else if ("elasticnet".equalsIgnoreCase(regName)) {
            return new ElasticNet(options);
        } else if ("rda".equalsIgnoreCase(regName)) {
            // Return `PassThrough` because we need special handling for RDA.
            // See an implementation of `Optimizer#RDA`.
            return new PassThrough(options);
        } else {
            throw new IllegalArgumentException("Unsupported regularization name: " + regName);
        }
    }

}
