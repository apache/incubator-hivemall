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
    private double meanRating;

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
        this.meanRating = 0.d;
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

    public double getMeanRating() {
        return meanRating;
    }

    public void setMeanRating(final double value) {
        meanRating = value;
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
//        RealMatrix BTB = computeWeightsTWeights(beta, factor, c0);
        RealMatrix BTB = computeWeightsTWeights(beta, factor, c0);
        if (identity == null) {
            identity = initIdentity(factor, lambdaTheta);
        }

        RealMatrix BTBpR = BTB.add(identity);

        String itemName;

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            // filter for trainable items
            List<Feature> trainableItems = new ArrayList<>();
            for (Feature item : sample.children) {
                itemName = item.getFeature();
                if (isTrainable(itemName, true)) {
                    trainableItems.add(item);
                }
            }

            RealMatrix A = calculateA(trainableItems);

            RealMatrix delta = calculateDelta(trainableItems, beta, factor, c1 - c0);
            RealMatrix B = BTBpR.add(delta);

            // solve and update factors
            RealMatrix newThetaMatrix = solve(B, A);
            setFactorVector(sample.parent.getFeature(), theta, newThetaMatrix.getRowVector(0));
        }
    }

    protected static RealMatrix initIdentity(int numFactors, float lambdaTheta) {
        RealMatrix I = new Array2DRowRealMatrix(MatrixUtils.eye(numFactors));
        I = I.scalarMultiply(lambdaTheta);
        return I;
    }

    private RealMatrix solve(RealMatrix b, RealMatrix a) {
        // b * x = a
        // solves for x
        SingularValueDecomposition svd = new SingularValueDecomposition(b);
        return svd.getSolver().solve(a);

    }

    private static RealMatrix calculateDelta(List<Feature> children, Map<String, RealVector> weights, int numFactors, float constant) {
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
            double rating = f.getValue();
            RealVector vec = getFactorVector(f.getFeature(), weights);
            result += rating * vec.getEntry(dim1) * vec.getEntry(dim2);
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

    private RealMatrix calculateA(List<Feature> items) {
        // Equivalent to: a = x_u.dot(c1 * B_u)
        // x_u is a (1, i) matrix of all ones
        // B_u is a (i, F) matrix
        // What it does: sums factor n of each item in B_u
        RealVector v = new ArrayRealVector(items.size());
        for (Feature item : items) {
            v.add(getBetaVector(item.getFeature()));
        }
        for (int a = 0; a < v.getDimension(); a++) {
            v.setEntry(a, v.getEntry(a) / c1);
        }
        // convert RealVector to 1-row RealMatrix
        RealMatrix A = new Array2DRowRealMatrix(1, items.size());
        A.setRowVector(0, v);
        return A;
    }

    protected static RealMatrix computeWeightsTWeights(Map<String, RealVector> weights, int numFactors, float constant) {
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


    protected static void accumulateInMatrix(RealVector u, RealVector v, RealMatrix BTB, float c0) {
        for (int i = 0; i < u.getDimension(); i++) {
            double val = u.getEntry(i);
            for (int j = 0; j < v.getDimension(); j++) {
                double newEntry = BTB.getEntry(i, j) + c0 * val * v.getEntry(j);
                BTB.setEntry(i, j, newEntry);
            }
        }
    }

    private boolean isTrainable(String name, boolean isItem) {
        return isItem ? beta.containsKey(name) : theta.containsKey(name);
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
