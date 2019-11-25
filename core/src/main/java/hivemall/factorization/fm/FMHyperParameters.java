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
package hivemall.factorization.fm;

import hivemall.factorization.fm.FactorizationMachineModel.VInitScheme;
import hivemall.optimizer.EtaEstimator;
import hivemall.optimizer.EtaEstimator.InvscalingEtaEstimator;
import hivemall.utils.lang.Primitives;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;

class FMHyperParameters {
    protected static final float DEFAULT_ETA0 = 0.1f;
    protected static final float DEFAULT_LAMBDA = 0.0001f;

    // -------------------------------------
    // Model parameters

    boolean classification = false;
    int factors = 5;

    // regularization
    float lambda = DEFAULT_LAMBDA;
    float lambdaW0;
    float lambdaW;
    float lambdaV;

    // V initialization
    double sigma = 0.1d;
    long seed = -1L;
    @Nonnull
    VInitScheme vInit;

    // regression
    double minTarget = Double.MIN_VALUE;
    double maxTarget = Double.MAX_VALUE;

    // learning rate
    @Nonnull
    EtaEstimator eta;

    // feature hashing
    int numFeatures = -1;

    // -------------------------------------
    // non-model parameters

    boolean l2norm; // enable by default for FFM. disabled by default for FM.

    int iters = 10;
    boolean conversionCheck = true;
    double convergenceRate = 0.005d;

    boolean earlyStopping = false;

    // adaptive regularization
    boolean adaptiveRegularization = false;
    float validationRatio = 0.05f;
    int validationThreshold = 1000;
    boolean parseFeatureAsInt = false;

    FMHyperParameters() {
        this.vInit = instantiateVInit();
        this.eta = new InvscalingEtaEstimator(DEFAULT_ETA0, EtaEstimator.DEFAULT_POWER_T);
    }

    @Override
    public String toString() {
        return "FMHyperParameters [classification=" + classification + ", factors=" + factors
                + ", lambda=" + lambda + ", lambdaW0=" + lambdaW0 + ", lambdaW=" + lambdaW
                + ", lambdaV=" + lambdaV + ", sigma=" + sigma + ", seed=" + seed + ", vInit="
                + vInit + ", minTarget=" + minTarget + ", maxTarget=" + maxTarget + ", eta=" + eta
                + ", numFeatures=" + numFeatures + ", l2norm=" + l2norm + ", iters=" + iters
                + ", conversionCheck=" + conversionCheck + ", convergenceRate=" + convergenceRate
                + ", adaptiveRegularization=" + adaptiveRegularization + ", validationRatio="
                + validationRatio + ", validationThreshold=" + validationThreshold
                + ", parseFeatureAsInt=" + parseFeatureAsInt + "]";
    }

    void processOptions(@Nonnull CommandLine cl) throws UDFArgumentException {
        this.classification = cl.hasOption("classification");
        if (cl.hasOption("factor")) {
            this.factors = Primitives.parseInt(cl.getOptionValue("factor"), factors);
        } else {
            this.factors = Primitives.parseInt(cl.getOptionValue("factors"), factors);
        }
        this.lambda = Primitives.parseFloat(cl.getOptionValue("lambda"), lambda);
        this.lambdaW0 = Primitives.parseFloat(cl.getOptionValue("lambda_w0"), lambda);
        this.lambdaW = Primitives.parseFloat(cl.getOptionValue("lambda_wi"), lambda);
        this.lambdaV = Primitives.parseFloat(cl.getOptionValue("lambda_v"), lambda);
        this.sigma = Primitives.parseDouble(cl.getOptionValue("sigma"), sigma);
        this.seed = Primitives.parseLong(cl.getOptionValue("seed"), seed);
        if (seed == -1L) {
            this.seed = System.nanoTime();
        }
        this.vInit = instantiateVInit(cl, factors, seed, classification);
        this.minTarget = Primitives.parseDouble(cl.getOptionValue("min_target"), minTarget);
        this.maxTarget = Primitives.parseDouble(cl.getOptionValue("max_target"), maxTarget);
        this.eta = EtaEstimator.get(cl, DEFAULT_ETA0);
        this.numFeatures = Primitives.parseInt(cl.getOptionValue("num_features"), numFeatures);
        this.l2norm = cl.hasOption("enable_norm");
        if (cl.hasOption("iter")) {
            this.iters = Primitives.parseInt(cl.getOptionValue("iter"), iters);
        } else {
            this.iters = Primitives.parseInt(cl.getOptionValue("iterations"), iters);
        }
        this.conversionCheck = !cl.hasOption("disable_cvtest");
        this.convergenceRate =
                Primitives.parseDouble(cl.getOptionValue("cv_rate"), convergenceRate);
        this.earlyStopping = cl.hasOption("early_stopping");
        this.adaptiveRegularization = cl.hasOption("adaptive_regularization");
        this.validationRatio =
                Primitives.parseFloat(cl.getOptionValue("validation_ratio"), validationRatio);
        if (validationRatio < 0.f || validationRatio >= 1.f) {
            throw new UDFArgumentException(
                "validation_ratio should be in range [0, 1): " + validationRatio);
        }
        this.validationThreshold =
                Primitives.parseInt(cl.getOptionValue("validation_threshold"), validationThreshold);
        this.parseFeatureAsInt = cl.hasOption("int_feature");
    }

    @Nonnull
    private VInitScheme instantiateVInit() {
        VInitScheme vInit = getDefaultVinitScheme(classification);
        vInit.setMaxInitValue(0.5f);
        vInit.setInitStdDev(0.2d);
        vInit.initRandom(factors, System.nanoTime());
        return vInit;
    }

    @Nonnull
    private VInitScheme instantiateVInit(@Nonnull CommandLine cl, int factor, long seed,
            final boolean classification) {
        String vInitOpt = cl.getOptionValue("init_v");
        float maxInitValue = Primitives.parseFloat(cl.getOptionValue("max_init_value"), 0.5f);
        double initStdDev = Primitives.parseDouble(cl.getOptionValue("min_init_stddev"), 0.1d);

        VInitScheme vInit = VInitScheme.resolve(vInitOpt, getDefaultVinitScheme(classification));
        vInit.setMaxInitValue(maxInitValue);
        initStdDev = Math.max(initStdDev, 1.0d / factor);
        vInit.setInitStdDev(initStdDev);
        vInit.initRandom(factor, seed);
        return vInit;
    }

    @Nonnull
    protected VInitScheme getDefaultVinitScheme(boolean classification) {
        return classification ? VInitScheme.gaussian : VInitScheme.adjustedRandom;
    }

    public static final class FFMHyperParameters extends FMHyperParameters {

        // FFM hyper parameters
        boolean globalBias = false;
        boolean linearCoeff = false;

        // feature hashing
        int numFields = Feature.DEFAULT_NUM_FIELDS;

        // adagrad
        boolean useAdaGrad = false;
        float eps = 1.f;

        // FTRL
        boolean useFTRL = false;
        float alphaFTRL = 0.5f; // Learning Rate
        float betaFTRL = 1.0f; // Smoothing parameter for AdaGrad
        float lambda1 = 0.0002f; // L1 Regularization
        float lambda2 = 0.0001f; // L2 Regularization

        FFMHyperParameters() {
            super();
        }

        @Nonnull
        protected VInitScheme getDefaultVinitScheme(boolean classification) {
            return VInitScheme.random;
        }

        @Override
        void processOptions(@Nonnull CommandLine cl) throws UDFArgumentException {
            super.processOptions(cl);

            if (cl.hasOption("int_feature")) {
                throw new UDFArgumentException("int_feature option is not supported yet for FFM");
            }

            this.globalBias = cl.hasOption("global_bias");
            this.linearCoeff = cl.hasOption("linear_term");

            if (cl.hasOption("enable_norm") && cl.hasOption("disable_norm")) {
                throw new UDFArgumentException(
                    "-enable_norm and -disable_norm MUST NOT be used simultaneously");
            }
            this.l2norm = !cl.hasOption("disable_norm");

            // feature hashing
            if (numFeatures == -1) {
                int hashbits = Primitives.parseInt(cl.getOptionValue("feature_hashing"), -1);
                if (hashbits != -1) {
                    if (hashbits < 18 || hashbits > 31) {
                        throw new UDFArgumentException(
                            "-feature_hashing MUST be in range [18,31]: " + hashbits);
                    }
                    this.numFeatures = 1 << hashbits;
                }
            }
            this.numFields = Primitives.parseInt(cl.getOptionValue("num_fields"), numFields);
            if (numFields <= 1) {
                throw new UDFArgumentException("-num_fields MUST be greater than 1: " + numFields);
            }

            // optimizer
            final String optimizer = cl.getOptionValue("optimizer", "ftrl").toLowerCase();
            switch (optimizer) {
                case "ftrl": {
                    this.useFTRL = true;
                    this.useAdaGrad = false;
                    this.alphaFTRL =
                            Primitives.parseFloat(cl.getOptionValue("alphaFTRL"), alphaFTRL);
                    if (alphaFTRL == 0.f) {
                        throw new UDFArgumentException("-alphaFTRL SHOULD NOT be 0");
                    }
                    this.betaFTRL = Primitives.parseFloat(cl.getOptionValue("betaFTRL"), betaFTRL);
                    this.lambda1 = Primitives.parseFloat(cl.getOptionValue("lambda1"), lambda1);
                    this.lambda2 = Primitives.parseFloat(cl.getOptionValue("lambda2"), lambda2);
                    break;
                }
                case "adagrad": {
                    this.useAdaGrad = true;
                    this.useFTRL = false;
                    this.eps = Primitives.parseFloat(cl.getOptionValue("eps"), eps);
                    break;
                }
                case "sgd":
                    // fall through
                default: {
                    this.useFTRL = false;
                    this.useAdaGrad = false;
                    break;
                }
            }
        }

        @Override
        public String toString() {
            return "FFMHyperParameters [globalBias=" + globalBias + ", linearCoeff=" + linearCoeff
                    + ", numFields=" + numFields + ", useAdaGrad=" + useAdaGrad + ", eps=" + eps
                    + ", useFTRL=" + useFTRL + ", alphaFTRL=" + alphaFTRL + ", betaFTRL=" + betaFTRL
                    + ", lambda1=" + lambda1 + ", lambda2=" + lambda2 + "], " + super.toString();
        }

    }

}
