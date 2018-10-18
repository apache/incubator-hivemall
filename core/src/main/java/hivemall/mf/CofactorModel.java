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

    public void recordContext(Feature context, Boolean isParentAnItem) {
        String key = context.getFeature();
        if (isParentAnItem) {
            initFactorVector(key, beta);
            initFactorVector(key, gamma);
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

    public void updateWithUsers(List<CofactorizationUDTF.TrainingSample> users) {
        updateTheta(users);
    }

    public void updateWithItems(List<CofactorizationUDTF.TrainingSample> items) {
        updateBeta(items);
        updateGamma(items);
        updateBetaBias(items);
        updateGammaBias(items);
    }

    /**
     * Update latent factors of the users in the provided mini-batch.
     */
    private void updateTheta(List<CofactorizationUDTF.TrainingSample> samples) {
        // initialize item factors
        // items should only be trainable if the dataset contains a major entry for that item (which it may not)

        // variable names follow cofacto.py
        RealMatrix BTBpR = calculateWTWpR(beta, factor, c0, identity, lambdaTheta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newThetaVec = calculateNewThetaVector(sample, beta, factor, BTBpR, c0, c1);
            if (newThetaVec != null) {
                setFactorVector(sample.context.getFeature(), theta, newThetaVec);
            }
        }
    }

    protected static RealVector calculateNewThetaVector(CofactorizationUDTF.TrainingSample sample, Map<String, RealVector> beta,
                                                        int numFactors, RealMatrix BTBpR, float c0, float c1) {
        // filter for trainable items
        List<Feature> trainableItems = filterTrainableFeatures(sample.features, beta);
        // TODO: is this correct behaviour?
        if (trainableItems.isEmpty()) {
            return null;
        }

        RealVector A = calculateA(trainableItems, beta, numFactors, c1);

        RealMatrix delta = calculateWTWSubset(trainableItems, beta, numFactors, c1 - c0);
        RealMatrix B = BTBpR.add(delta);

        // solve and update factors
        RealVector newThetaVec = solve(B, A);
        return newThetaVec;
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateBeta(List<CofactorizationUDTF.TrainingSample> samples) {
        // precomputed matrix
        RealMatrix TTTpR = calculateWTWpR(theta, factor, c0, identity, lambdaBeta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newBetaVec = calculateNewBetaVector(sample, theta, gamma, gammaBias, betaBias, factor, TTTpR, c0, c1);
            if (newBetaVec != null) {
                setFactorVector(sample.context.getFeature(), beta, newBetaVec);
            }
        }
    }

    protected static RealVector calculateNewBetaVector(CofactorizationUDTF.TrainingSample sample, Map<String, RealVector> theta,
                                                       Map<String, RealVector> gamma, Map<String, Double> gammaBias,
                                                       Map<String, Double> betaBias, int numFactors, RealMatrix TTTpR, float c0, float c1) {
        // filter for trainable users
        List<Feature> trainableUsers = filterTrainableFeatures(sample.features, theta);
        // TODO: is this correct behaviour?
        if (trainableUsers.isEmpty()) {
            return null;
        }

        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, gamma);
        RealVector RSD = calculateRSD(sample.context, trainableCooccurringItems, numFactors, betaBias, gammaBias, gamma);
        RealVector ApRSD = calculateA(trainableUsers, theta, numFactors, c1).add(RSD);

        RealMatrix GTG = calculateWTWSubset(trainableCooccurringItems, gamma, numFactors, 1.f);
        RealMatrix delta = calculateWTWSubset(trainableUsers, theta, numFactors, c1 - c0);
        RealMatrix B = TTTpR.add(delta).add(GTG);

        // solve and update factors
        RealVector newBetaVec = solve(B, ApRSD);
        return newBetaVec;
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateGamma(List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newGammaVec = calculateNewGammaVector(sample, beta, gammaBias, betaBias, factor, identity, lambdaGamma);
            if (newGammaVec != null) {
                setFactorVector(sample.context.getFeature(), gamma, newGammaVec);
            }
        }
    }

    protected static RealVector calculateNewGammaVector(CofactorizationUDTF.TrainingSample sample, Map<String, RealVector> beta,
                                                      Map<String, Double> gammaBias, Map<String, Double> betaBias,
                                                      int numFactors, RealMatrix idMatrix, float lambdaGamma) {
        // filter for trainable items
        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        // TODO: is this correct behaviour?
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }

        RealMatrix B = calculateWTWSubset(trainableCooccurringItems, beta, numFactors, 1.f)
                .add(calculateR(idMatrix, lambdaGamma, numFactors));
        RealVector rsd = calculateRSD(sample.context, trainableCooccurringItems, numFactors, gammaBias, betaBias, beta);

        // solve and update factors
        RealVector newGammaVec = solve(B, rsd);
        return newGammaVec;
    }

    private void updateBetaBias(List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newBetaBias = calculateNewBias(sample, beta, gamma, gammaBias);
            // TODO: is this correct behaviour?
            if (newBetaBias != null) {
                setBetaBias(sample.context.getFeature(), newBetaBias);
            }
        }
    }

    public void updateGammaBias(List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newGammaBias = calculateNewBias(sample, gamma, beta, betaBias);
            // TODO: is this correct behaviour?
            if (newGammaBias != null) {
                setBetaBias(sample.context.getFeature(), newGammaBias);
            }
        }
    }

    protected static Double calculateNewBias(CofactorizationUDTF.TrainingSample sample, Map<String, RealVector> beta,
                                               Map<String, RealVector> gamma, Map<String, Double> biases) {
        // filter for trainable items
        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }

        double rsd = calculateBiasRSD(sample.context, trainableCooccurringItems, beta, gamma, biases);
        return rsd / trainableCooccurringItems.size();

    }

    protected static double calculateBiasRSD(Feature thisItem, List<Feature> trainableItems, Map<String, RealVector> beta,
                                           Map<String, RealVector> gamma, Map<String, Double> biases) {
        double result = 0.d;
        String i = thisItem.getFeature();
        RealVector thisFactorVec = getFactorVector(i, beta);

        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            double value = cooccurrence.getValue() - thisFactorVec.dotProduct(getFactorVector(j, gamma)) - getBias(j, biases);
            result += value;
        }
        return result;
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

    protected static RealMatrix calculateR(RealMatrix idMatrix, float lambda, int numFactors) {
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

    protected static RealMatrix calculateWTWSubset(List<Feature> subset, Map<String, RealVector> weights, int numFactors, float constant) {
        // equivalent to `B_u.T.dot((c1 - c0) * B_u)` in cofacto.py
        RealMatrix delta = new Array2DRowRealMatrix(numFactors, numFactors);
        int i = 0, j = 0;
        for (int f = 0; f < numFactors; f++) {
            for (int ff = 0; ff < numFactors; ff++) {
                double val = constant * dotFactorsAlongDims(subset, weights, f, ff);
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

    protected static RealVector calculateA(List<Feature> items, Map<String, RealVector> weights, int numFactors, float constant) {
        // Equivalent to: a = x_u.dot(c1 * B_u)
        // x_u is a (1, i) matrix of all ones
        // B_u is a (i, F) matrix
        // What it does: sums factor n of each item in B_u
        RealVector v = new ArrayRealVector(numFactors);
        for (Feature item : items) {
            double y_ui = item.getValue(); // rating
            addInPlace(v, getFactorVector(item.getFeature(), weights), y_ui);
        }
        for (int a = 0; a < v.getDimension(); a++) {
            v.setEntry(a, v.getEntry(a) * constant);
        }
        return v;
    }

    public Double calculateLoss(List<CofactorizationUDTF.TrainingSample> users, List<CofactorizationUDTF.TrainingSample> items) {
        // for speed - can calculate loss on a small subset of the training data
        double mf_loss = calculateMFLoss(users, theta, beta, c0, c1) + calculateMFLoss(items, beta, theta, c0, c1);
        double embed_loss = calculateEmbedLoss(items, beta, gamma, betaBias, gammaBias);
        return mf_loss + embed_loss + sumL2Loss(theta, lambdaTheta) + sumL2Loss(beta, lambdaBeta) + sumL2Loss(gamma, lambdaGamma);

    }

    protected static double calculateEmbedLoss(List<CofactorizationUDTF.TrainingSample> items, Map<String, RealVector> beta,
                                               Map<String, RealVector> gamma, Map<String, Double> betaBias,
                                               Map<String, Double> gammaBias) {
        double loss = 0.d, val, bBias, gBias;
        RealVector bFactors, gFactors;
        String bKey, gKey;
        for (CofactorizationUDTF.TrainingSample item: items) {
            bKey = item.context.getFeature();
            bFactors = getFactorVector(bKey, beta);
            bBias = getBias(bKey, betaBias);
            for (Feature cooccurrence : item.sppmi) {
                if (!isTrainable(cooccurrence.getFeature(), beta)) {
                    continue;
                }
                gKey = cooccurrence.getFeature();
                gFactors = getFactorVector(gKey, gamma);
                gBias = getBias(gKey, gammaBias);
                val = cooccurrence.getValue() - bFactors.dotProduct(gFactors) - bBias - gBias;
                loss += val * val;
            }
        }
        return loss;
    }

    protected static double calculateMFLoss(List<CofactorizationUDTF.TrainingSample> samples, Map<String, RealVector> contextWeights,
                                            Map<String, RealVector> featureWeights, float c0, float c1) {
        double loss = 0.d, err, predicted, y;
        RealVector contextFactors, ratedFactors;

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            contextFactors = getFactorVector(sample.context.getFeature(), contextWeights);
            // all items / users
            for (RealVector unratedFactors : featureWeights.values()) {
                predicted = contextFactors.dotProduct(unratedFactors);
                err = (0.d - predicted);
                loss += c0 * err * err;
            }
            // only rated items / users
            for (Feature f : sample.features) {
                if (!isTrainable(f.getFeature(), featureWeights)) {
                    continue;
                }
                ratedFactors = getFactorVector(f.getFeature(), featureWeights);
                predicted = contextFactors.dotProduct(ratedFactors);
                y = f.getValue();
                err = y - predicted;
                loss += (c1 - c0) * err * err;
            }
        }
        return loss;
    }

    protected static double sumL2Loss(Map<String, RealVector> weights, float lambda) {
        double loss = 0.d;
        for (RealVector v : weights.values()) {
            loss += L2Distance(v);
        }
        return lambda * loss;
    }

    protected static double L2Distance(RealVector v) {
        double result = 0.d;
        for (int i = 0; i < v.getDimension(); i++) {
            double val = v.getEntry(i);
            result +=  val * val;
        }
        return Math.sqrt(result);
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
