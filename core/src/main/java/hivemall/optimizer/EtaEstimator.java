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

import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.StringUtils;

import java.util.Map;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;

public abstract class EtaEstimator {

    public static final float DEFAULT_ETA0 = 0.1f;
    public static final float DEFAULT_ETA = 0.3f;
    public static final double DEFAULT_POWER_T = 0.1d;

    protected final float eta0;

    public EtaEstimator(float eta0) {
        this.eta0 = eta0;
    }

    @Nonnull
    public abstract String typeName();

    public float eta0() {
        return eta0;
    }

    public abstract float eta(long t);

    public void update(@Nonnegative float multiplier) {}

    public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
        hyperParams.put("eta", typeName());
        hyperParams.put("eta0", eta0());
    }

    public static final class FixedEtaEstimator extends EtaEstimator {

        public FixedEtaEstimator(float eta) {
            super(eta);
        }

        @Nonnull
        public String typeName() {
            return "Fixed";
        }

        @Override
        public float eta(long t) {
            return eta0;
        }

        @Override
        public String toString() {
            return "FixedEtaEstimator [ eta0 = " + eta0 + " ]";
        }

    }

    public static final class SimpleEtaEstimator extends EtaEstimator {

        private final float finalEta;
        private final double total_steps;

        public SimpleEtaEstimator(float eta0, long total_steps) {
            super(eta0);
            this.finalEta = (float) (eta0 / 2.d);
            this.total_steps = total_steps;
        }

        @Nonnull
        public String typeName() {
            return "Simple";
        }

        @Override
        public float eta(final long t) {
            if (t > total_steps) {
                return finalEta;
            }
            return (float) (eta0 / (1.d + (t / total_steps)));
        }

        @Override
        public String toString() {
            return "SimpleEtaEstimator [ eta0 = " + eta0 + ", totalSteps = " + total_steps
                    + ", finalEta = " + finalEta + " ]";
        }

        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("total_steps", total_steps);
        }

    }

    public static final class InvscalingEtaEstimator extends EtaEstimator {

        private final double power_t;

        public InvscalingEtaEstimator(float eta0, double power_t) {
            super(eta0);
            this.power_t = power_t;
        }

        @Nonnull
        public String typeName() {
            return "Invscaling";
        }

        @Override
        public float eta(final long t) {
            return (float) (eta0 / Math.pow(t, power_t));
        }

        @Override
        public String toString() {
            return "InvscalingEtaEstimator [ eta0 = " + eta0 + ", power_t = " + power_t + " ]";
        }

        public void getHyperParameters(@Nonnull Map<String, Object> hyperParams) {
            super.getHyperParameters(hyperParams);
            hyperParams.put("power_t", power_t);
        }
    }

    /**
     * bold driver: Gemulla et al., Large-scale matrix factorization with distributed stochastic
     * gradient descent, KDD 2011.
     */
    public static final class AdjustingEtaEstimator extends EtaEstimator {

        private float eta;

        public AdjustingEtaEstimator(float eta) {
            super(eta);
            this.eta = eta;
        }

        @Nonnull
        public String typeName() {
            return "boldDriver";
        }

        @Override
        public float eta(long t) {
            return eta;
        }

        @Override
        public void update(@Nonnegative float multiplier) {
            float newEta = eta * multiplier;
            if (!NumberUtils.isFinite(newEta)) {
                // avoid NaN or INFINITY
                return;
            }
            this.eta = Math.min(eta0, newEta); // never be larger than eta0
        }

        @Override
        public String toString() {
            return "AdjustingEtaEstimator [ eta0 = " + eta0 + ", eta = " + eta + " ]";
        }

    }

    @Nonnull
    public static EtaEstimator get(@Nullable CommandLine cl) throws UDFArgumentException {
        return get(cl, DEFAULT_ETA0);
    }

    @Nonnull
    public static EtaEstimator get(@Nullable CommandLine cl, float defaultEta0)
            throws UDFArgumentException {
        if (cl == null) {
            return new InvscalingEtaEstimator(defaultEta0, DEFAULT_POWER_T);
        }

        if (cl.hasOption("boldDriver")) {
            float eta = Primitives.parseFloat(cl.getOptionValue("eta"), DEFAULT_ETA);
            return new AdjustingEtaEstimator(eta);
        }

        String etaValue = cl.getOptionValue("eta");
        if (etaValue != null) {
            float eta = Float.parseFloat(etaValue);
            return new FixedEtaEstimator(eta);
        }

        float eta0 = Primitives.parseFloat(cl.getOptionValue("eta0"), defaultEta0);
        if (cl.hasOption("t")) {
            long t = Long.parseLong(cl.getOptionValue("t"));
            return new SimpleEtaEstimator(eta0, t);
        }

        double power_t = Primitives.parseDouble(cl.getOptionValue("power_t"), DEFAULT_POWER_T);
        return new InvscalingEtaEstimator(eta0, power_t);
    }

    @Nonnull
    public static EtaEstimator get(@Nonnull final Map<String, String> options)
            throws IllegalArgumentException {
        final float eta0 = Primitives.parseFloat(options.get("eta0"), DEFAULT_ETA0);
        final double power_t = Primitives.parseDouble(options.get("power_t"), DEFAULT_POWER_T);

        final String etaScheme = options.get("eta");
        if (etaScheme == null) {
            return new InvscalingEtaEstimator(eta0, power_t);
        }

        if ("fixed".equalsIgnoreCase(etaScheme)) {
            return new FixedEtaEstimator(eta0);
        } else if ("simple".equalsIgnoreCase(etaScheme)) {
            final long t;
            if (options.containsKey("total_steps")) {
                t = Long.parseLong(options.get("total_steps"));
            } else {
                throw new IllegalArgumentException(
                    "-total_steps MUST be provided when `-eta simple` is specified");
            }
            return new SimpleEtaEstimator(eta0, t);
        } else if ("inv".equalsIgnoreCase(etaScheme) || "inverse".equalsIgnoreCase(etaScheme)
                || "invscaling".equalsIgnoreCase(etaScheme)) {
            return new InvscalingEtaEstimator(eta0, power_t);
        } else {
            if (StringUtils.isNumber(etaScheme)) {
                float eta = Float.parseFloat(etaScheme);
                return new FixedEtaEstimator(eta);
            }
            throw new IllegalArgumentException("Unsupported ETA name: " + etaScheme);
        }
    }

}
