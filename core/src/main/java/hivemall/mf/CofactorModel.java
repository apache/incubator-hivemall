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
package hivemall.mf;

import hivemall.fm.Feature;
import hivemall.utils.math.MathUtils;
import hivemall.utils.math.MatrixUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;

public class CofactorModel {

    public enum RankInitScheme {
        random /* default */, gaussian;

        @Nonnegative
        protected float maxInitValue;
        @Nonnegative
        protected double initStdDev;

        @Nonnull
        public static CofactorModel.RankInitScheme resolve(@Nullable String opt) {
            if (opt == null) {
                return random;
            } else if ("gaussian".equalsIgnoreCase(opt)) {
                return gaussian;
            } else if ("random".equalsIgnoreCase(opt)) {
                return random;
            }
            return random;
        }

        public void setMaxInitValue(float maxInitValue) {
            this.maxInitValue = maxInitValue;
        }

        public void setInitStdDev(double initStdDev) {
            this.initStdDev = initStdDev;
        }

    }

    private static final int EXPECTED_SIZE = 136861;
    @Nonnegative
    protected final int factor;

    // rank matrix initialization
    protected final RankInitScheme initScheme;

    @Nonnull
    private double globalBias;

    // storing trainable latent factors and weights
    private Map<String, RealVector> theta;
    private Map<String, RealVector> beta;
    private Map<String, Double> betaBias;
    private Map<String, RealVector> gamma;
    private Map<String, Double> gammaBias;

    // precomputed identity matrix
    private RealMatrix identity;

    protected final Random[] randU, randI;

    // hyperparameters
    private final float c0, c1;
    private final float lambdaTheta, lambdaBeta, lambdaGamma;

    public CofactorModel(@Nonnegative int factor, @Nonnull RankInitScheme initScheme,
                         @Nonnull float c0, @Nonnull float c1, float lambdaTheta,
                         float lambdaBeta, float lambdaGamma) {

        // rank init scheme is gaussian
        // https://github.com/dawenl/cofactor/blob/master/src/cofacto.py#L98
        this.factor = factor;
        this.initScheme = initScheme;
        this.globalBias = 0.d;
        this.lambdaTheta = lambdaTheta;
        this.lambdaBeta = lambdaBeta;
        this.lambdaGamma = lambdaGamma;

        this.theta = new HashMap<>();
        this.beta = new HashMap<>();
        this.betaBias = new HashMap<>();
        this.gamma = new HashMap<>();
        this.gammaBias = new HashMap<>();

        this.randU = newRandoms(factor, 31L);
        this.randI = newRandoms(factor, 41L);

        checkHyperparameterC(c0);
        checkHyperparameterC(c1);
        this.c0 = c0;
        this.c1 = c1;

    }

    private void initFactorVector(String key, Map<String, RealVector> weights) {
        if (weights.containsKey(key)) {
            return;
        }
        RealVector v = new ArrayRealVector(factor);
        switch (initScheme) {
            case random:
                uniformFill(v, randI[0], initScheme.maxInitValue);
                break;
            case gaussian:
                gaussianFill(v, randI, initScheme.initStdDev);
                break;
            default:
                throw new IllegalStateException(
                        "Unsupported rank initialization scheme: " + initScheme);

        }
        weights.put(key, v);
    }

    private static RealVector getFactorVector(String key, Map<String, RealVector> weights) {
        return weights.get(key);
    }

    private static void setFactorVector(String key, Map<String, RealVector> weights, RealVector factorVector) {
        assert weights.containsKey(key);
        weights.put(key, factorVector);
    }

    private static double getBias(String key, Map<String, Double> biases) {
        if (!biases.containsKey(key)) {
            biases.put(key, 0.d);
        }
        return biases.get(key);
    }

    private static void setBias(String key, Map<String, Double> biases, double value) {
        biases.put(key, value);
    }

    public void recordAsParent(Feature parent, Boolean isParentAnItem) {
        String key = parent.getFeature();
        if (isParentAnItem) {
            initFactorVector(key, beta);
            initFactorVector(key, gamma);
            getBetaBias(key);
            getGammaBias(key);
        } else {
            initFactorVector(key, theta);
        }
    }

    public RealVector getGammaVector(final String key) {
        return getFactorVector(key, gamma);
    }

    public double getGammaBias(final String key) {
        return getBias(key, gammaBias);
    }

    public void setGammaBias(final String key, final double value) {
        setBias(key, gammaBias, value);
    }

    public double getGlobalBias() {
        return globalBias;
    }

    public void setGlobalBias(final double value) {
        globalBias = value;
    }

    public RealVector getThetaVector(final String key) {
        return getFactorVector(key, theta);
    }

    public RealVector getBetaVector(final String key) {
        return getFactorVector(key, beta);
    }

    public double getBetaBias(final String key) {
        return getBias(key, betaBias);
    }

    public void setBetaBias(final String key, final double value) {
        setBias(key, betaBias, value);
    }

    /**
     * Update latent factors of the users in the provided mini-batch.
     */
    public void updateTheta(List<CofactorizationUDTF.TrainingSample> samples) {
        // initialize item factors
        // items should only be trainable if the dataset contains a major entry for that item (which it may not)

        // variable names follow cofacto.py
        RealMatrix BTBpR = calculateWTWpR(beta, factor, c0, identity, lambdaTheta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            // filter for trainable items
            List<Feature> trainableItems = filterTrainableFeatures(sample.children, beta);
            // TODO: is this correct behaviour?
            if (trainableItems.isEmpty()) {
                continue;
            }

            RealVector A = calculateA(trainableItems, beta, c1);

            RealMatrix delta = calculateDelta(trainableItems, beta, factor, c1 - c0);
            RealMatrix B = BTBpR.add(delta);

            // solve and update factors
            RealVector newThetaVec = solve(B, A);
            setFactorVector(sample.parent.getFeature(), theta, newThetaVec);
        }
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    public void updateBeta(List<CofactorizationUDTF.TrainingSample> samples) {
        // variable names follow cofacto.py
        RealMatrix TTTpR = calculateWTWpR(theta, factor, c0, identity, lambdaBeta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            // filter for trainable users
            List<Feature> trainableUsers = filterTrainableFeatures(sample.children, theta);
            // TODO: is this correct behaviour?
            if (trainableUsers.isEmpty()) {
                continue;
            }

            List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmiVector, gamma);
            RealVector RSD = calculateRSD(sample.parent, trainableCooccurringItems, factor, betaBias, gammaBias, gamma);
            RealVector ApRSD = calculateA(trainableUsers, theta, c1).add(RSD);

            RealMatrix GTG = calculateDelta(trainableCooccurringItems, gamma, factor, 1.f);
            RealMatrix delta = calculateDelta(trainableUsers, theta, factor, c1 - c0);
            RealMatrix B = TTTpR.add(delta).add(GTG);

            // solve and update factors
            RealVector newBetaVec = solve(B, ApRSD);
            setFactorVector(sample.parent.getFeature(), beta, newBetaVec);
        }

    }

    protected static RealVector calculateRSD(Feature thisItem, List<Feature> trainableItems, int numFactors,
                                    Map<String, Double> fixedBias, Map<String, Double> changingBias, Map<String, RealVector> weights) {

        String i = thisItem.getFeature();
        double b = getBias(i, fixedBias);

        RealVector accumulator = new ArrayRealVector(numFactors);

        // m_ij is named the same as in cofacto.py
        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            double scale = cooccurrence.getValue() - b - getBias(j, changingBias);
            RealVector g = getFactorVector(j, weights);
            addInPlace(accumulator, g, scale);
        }
        return accumulator;
    }

    /**
     * Calculate W' x W plus regularization matrix
     */
    protected static RealMatrix calculateWTWpR(Map<String, RealVector> W, int numFactors, float c0, RealMatrix idMatrix, float lambda) {
        RealMatrix WTW = calculateWTW(W, numFactors, c0);
        RealMatrix R = calculateR(idMatrix, lambda, numFactors);
        return WTW.add(R);
    }

    private static RealMatrix calculateR(RealMatrix idMatrix, float lambda, int numFactors) {
        if (idMatrix == null) {
            idMatrix = new Array2DRowRealMatrix(MatrixUtils.eye(numFactors));
        }
        return idMatrix.scalarMultiply(lambda);
    }

    protected static List<Feature> filterTrainableFeatures(Feature[] features, Map<String, RealVector> weights) {
        List<Feature> trainableFeatures = new ArrayList<>();
        String fName;
        for (Feature f : features) {
            fName = f.getFeature();
            if (isTrainable(fName, weights)) {
                trainableFeatures.add(f);
            }
        }
        return trainableFeatures;
    }

    protected static RealVector solve(RealMatrix B, RealVector a) {
        // b * x = a
        // solves for x
        SingularValueDecomposition svd = new SingularValueDecomposition(B);
        return svd.getSolver().solve(a);

    }

    protected static RealMatrix calculateDelta(List<Feature> children, Map<String, RealVector> weights, int numFactors, float constant) {
        // equivalent to `B_u.T.dot((c1 - c0) * B_u)` in cofacto.py
        RealMatrix delta = new Array2DRowRealMatrix(numFactors, numFactors);
        int i = 0, j = 0;
        for (int f = 0; f < numFactors; f++) {
            for (int ff = 0; ff < numFactors; ff++) {
                double val = constant * dotFactorsAlongDims(children, weights, f, ff);
                delta.setEntry(f, ff, val);
            }
        }
        return delta;
    }

    private static double dotFactorsAlongDims(List<Feature> keys, Map<String, RealVector> weights, int dim1, int dim2) {
        double result = 0.d;
        for (Feature f : keys) {
            RealVector vec = getFactorVector(f.getFeature(), weights);
            result += vec.getEntry(dim1) * vec.getEntry(dim2);
        }
        return result;
    }

    private static double dotFactorsAlongDims(Map<String, RealVector> weights, int dim1, int dim2) {
        double result = 0.d;
        for (String key : weights.keySet()) {
            RealVector vec = getFactorVector(key, weights);
            result += vec.getEntry(dim1) * vec.getEntry(dim2);
        }
        return result;
    }

    protected static RealVector calculateA(List<Feature> items, Map<String, RealVector> weights, float constant) {
        // Equivalent to: a = x_u.dot(c1 * B_u)
        // x_u is a (1, i) matrix of all ones
        // B_u is a (i, F) matrix
        // What it does: sums factor n of each item in B_u
        RealVector v = new ArrayRealVector(items.size());
        for (Feature item : items) {
            double y_ui = item.getValue(); // rating
            addInPlace(v, getFactorVector(item.getFeature(), weights), y_ui);
        }
        for (int a = 0; a < v.getDimension(); a++) {
            v.setEntry(a, v.getEntry(a) * constant);
        }
        return v;
    }

    /**
     * Add v to u in-place without creating a new RealVector instance.
     * @param u vector to which v will be added
     * @param v vector containing new values to be added to u
     * @param scalar value to multiply each entry in v before adding to u
     */
    private static void addInPlace(RealVector u, RealVector v, double scalar) {
        assert u.getDimension() == v.getDimension();
        for (int i = 0; i < u.getDimension(); i++) {
            double newVal = u.getEntry(i) + scalar * v.getEntry(i);
            u.setEntry(i, newVal);
        }
    }

    protected static RealMatrix calculateWTW(Map<String, RealVector> weights, int numFactors, float constant) {
        RealMatrix WTW = new Array2DRowRealMatrix(numFactors, numFactors);
        int i = 0, j = 0;
        for (int f = 0; f < numFactors; f++) {
            for (int ff = 0; ff < numFactors; ff++) {
                double val = constant * dotFactorsAlongDims(weights, f, ff);
                WTW.setEntry(f, ff, val);
            }
        }
        return WTW;
    }

    private static boolean isTrainable(String name, Map<String, RealVector> weights) {
        return weights.containsKey(name);
    }

    @Nonnull
    private static Random[] newRandoms(@Nonnull final int size, final long seed) {
        final Random[] rand = new Random[size];
        for (int i = 0, len = rand.length; i < len; i++) {
            rand[i] = new Random(seed + i);
        }
        return rand;
    }

    protected static void uniformFill(final RealVector a, final Random rand, final float maxInitValue) {
        for (int i = 0, len = a.getDimension(); i < len; i++) {
            double v = rand.nextDouble() * maxInitValue / len;
            a.append(v);
        }
    }

    protected static void gaussianFill(final RealVector a, final Random[] rand, final double stddev) {
        for (int i = 0, len = a.getDimension(); i < len; i++) {
            double v = MathUtils.gaussian(0.d, stddev, rand[i]);
            a.append(v);
        }
    }

    private static void checkHyperparameterC(final float c) {
        assert c >= 0.f && c <= 1.f;
    }
}
