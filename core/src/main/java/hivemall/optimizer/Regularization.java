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

import javax.annotation.Nonnull;
import java.util.Map;

public abstract class Regularization {
    /** the default regularization term 0.0001 */
    public static final float DEFAULT_LAMBDA = 0.0001f;

    protected final float lambda;

    public Regularization(@Nonnull Map<String, String> options) {
        float lambda = DEFAULT_LAMBDA;
        if(options.containsKey("lambda")) {
            lambda = Float.parseFloat(options.get("lambda"));
        }
        this.lambda = lambda;
    }

    abstract float regularize(float weight, float gradient);

    public static final class PassThrough extends Regularization {

        public PassThrough(final Map<String, String> options) {
            super(options);
        }

        @Override
        public float regularize(float weight, float gradient) {
            return gradient;
        }

    }

    public static final class L1 extends Regularization {

        public L1(Map<String, String> options) {
            super(options);
        }

        @Override
        public float regularize(float weight, float gradient) {
            return gradient + lambda * (weight > 0.f? 1.f : -1.f);
        }

    }

    public static final class L2 extends Regularization {

        public L2(final Map<String, String> options) {
            super(options);
        }

        @Override
        public float regularize(float weight, float gradient) {
            return gradient + lambda * weight;
        }

    }

    @Nonnull
    public static Regularization get(@Nonnull final Map<String, String> options)
            throws IllegalArgumentException {
        final String regName = options.get("regularization");
        if (regName == null) {
            return new PassThrough(options);
        }
        
        if(regName.toLowerCase().equals("no")) {
            return new PassThrough(options);
        } else if(regName.toLowerCase().equals("l1")) {
            return new L1(options);
        } else if(regName.toLowerCase().equals("l2")) {
            return new L2(options);
        } else if(regName.toLowerCase().equals("rda")) {
            // Return `PassThrough` because we need special handling for RDA.
            // See an implementation of `Optimizer#RDA`.
            return new PassThrough(options);
        } else {
            throw new IllegalArgumentException("Unsupported regularization name: " + regName);
        }
    }

}