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

import hivemall.annotations.VisibleForTesting;
import hivemall.fm.Feature;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.MathUtils;
import it.unimi.dsi.fastutil.objects.Object2DoubleArrayMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import org.apache.commons.math3.linear.*;
import org.apache.hadoop.hive.ql.metadata.HiveException;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class CofactorModel {

    public enum RankInitScheme {
        random /* default */, gaussian;
        @Nonnegative
        private float maxInitValue;
        @Nonnegative
        private double initStdDev;

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

    @Nonnegative
    private final int factor;

    // rank matrix initialization
    private final RankInitScheme initScheme;

    private double globalBias;

    // storing trainable latent factors and weights
    private final Map<String, double[]> theta;
    private final Map<String, double[]> beta;
    private final Object2DoubleMap<String> betaBias;
    private final Map<String, double[]> gamma;
    private final Object2DoubleMap<String> gammaBias;

    private final Random[] randU, randI;

    // hyperparameters
    @Nonnegative
    private final float c0, c1;
    private final float lambdaTheta, lambdaBeta, lambdaGamma;

    // solve
    private final RealMatrix B;
    private final RealVector A;

    // error message strings
    private static final String ARRAY_NOT_SQUARE_ERR = "Array is not square";
    private static final String DIFFERENT_DIMS_ERR = "Matrix, vector or array do not match in size";

    public CofactorModel(@Nonnegative final int factor, @Nonnull final RankInitScheme initScheme,
                         @Nonnegative final float c0, @Nonnegative final float c1, @Nonnegative final float lambdaTheta,
                         @Nonnegative final float lambdaBeta, @Nonnegative final float lambdaGamma, final float globalBias) {

        // rank init scheme is gaussian
        // https://github.com/dawenl/cofactor/blob/master/src/cofacto.py#L98
        this.factor = factor;
        this.initScheme = initScheme;
        this.globalBias = globalBias;
        this.lambdaTheta = lambdaTheta;
        this.lambdaBeta = lambdaBeta;
        this.lambdaGamma = lambdaGamma;

        this.theta = new HashMap<>();
        this.beta = new HashMap<>();
        this.betaBias = new Object2DoubleArrayMap<>();
        this.betaBias.defaultReturnValue(0.d);
        this.gamma = new HashMap<>();
        this.gammaBias = new Object2DoubleArrayMap<>();
        this.gammaBias.defaultReturnValue(0.d);

        this.B = new Array2DRowRealMatrix(this.factor, this.factor);
        this.A = new ArrayRealVector(this.factor);

        this.randU = newRandoms(factor, 31L);
        this.randI = newRandoms(factor, 41L);

        Preconditions.checkArgument(c0 >= 0.f && c0 <= 1.f);
        Preconditions.checkArgument(c1 >= 0.f && c1 <= 1.f);

        this.c0 = c0;
        this.c1 = c1;

    }

    private void initFactorVector(final String key, final Map<String, double[]> weights) throws HiveException {
        if (weights.containsKey(key)) {
            throw new HiveException(String.format("two items or two users cannot have same `context` in training set: found duplicate context `%s`", key));
        }
        final double[] v = new double[factor];
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

    @Nullable
    private static double[] getFactorVector(String key, Map<String, double[]> weights) {
        return weights.get(key);
    }

    private static void setFactorVector(final String key, final Map<String, double[]> weights, final RealVector factorVector) throws HiveException {
        final double[] vec = weights.get(key);
        if (vec == null) {
            throw new HiveException();
        }
        copyData(vec, factorVector);
    }

    private static double getBias(String key, Object2DoubleMap<String> biases) {
        return biases.getDouble(key);
    }

    private static void setBias(String key, Object2DoubleMap<String> biases, double value) {
        biases.put(key, value);
    }

    public void recordContext(String context, Boolean isItem) throws HiveException {
        if (isItem) {
            initFactorVector(context, beta);
            initFactorVector(context, gamma);
        } else {
            initFactorVector(context, theta);
        }
    }

    @Nullable
    public double[] getGammaVector(@Nonnull final String key) {
        return getFactorVector(key, gamma);
    }

    public double getGammaBias(@Nonnull final String key) {
        return getBias(key, gammaBias);
    }

    public void setGammaBias(@Nonnull final String key, final double value) {
        setBias(key, gammaBias, value);
    }

    public double getGlobalBias() {
        return globalBias;
    }

    public void setGlobalBias(final double value) {
        globalBias = value;
    }

    @Nullable
    public double[] getThetaVector(@Nonnull final String key) {
        return getFactorVector(key, theta);
    }

    @Nullable
    public double[] getBetaVector(@Nonnull final String key) {
        return getFactorVector(key, beta);
    }

    public double getBetaBias(@Nonnull final String key) {
        return getBias(key, betaBias);
    }

    public void setBetaBias(@Nonnull final String key, final double value) {
        setBias(key, betaBias, value);
    }

    @Nonnull
    public Map<String, double[]> getTheta() {
        return theta;
    }

    @Nonnull
    public Map<String, double[]> getBeta() {
        return beta;
    }

    @Nonnull
    public Map<String, double[]> getGamma() {
        return gamma;
    }

    @Nonnull
    public Object2DoubleMap<String> getBetaBiases() {
        return betaBias;
    }

    @Nonnull
    public Object2DoubleMap<String> getGammaBiases() {
        return gammaBias;
    }

    public void updateWithUsers(@Nonnull final List<CofactorizationUDTF.TrainingSample> users) throws HiveException {
        updateTheta(users);
    }

    public void updateWithItems(@Nonnull final List<CofactorizationUDTF.TrainingSample> items) throws HiveException {
        updateBeta(items);
        updateGamma(items);
        updateBetaBias(items);
        updateGammaBias(items);
        updateGlobalBias(items);
    }

    /**
     * Update latent factors of the users in the provided mini-batch.
     */
    private void updateTheta(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        // initialize item factors
        // items should only be trainable if the dataset contains a major entry for that item (which it may not)
        // variable names follow cofacto.py
        final double[][] BTBpR = calculateWTWpR(beta, factor, c0, lambdaTheta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newThetaVec = calculateNewThetaVector(sample, beta, factor, B, A, BTBpR, c0, c1);
            if (newThetaVec != null) {
                setFactorVector(sample.context, theta, newThetaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewThetaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Map<String, double[]> beta,
                                                        @Nonnegative final int numFactors, @Nonnull final RealMatrix B, @Nonnull final RealVector A,
                                                        @Nonnull final double[][] BTBpR, @Nonnegative final float c0, @Nonnegative final float c1) throws HiveException {
        // filter for trainable items
        List<Feature> trainableItems = filterTrainableFeatures(sample.features, beta);
        if (trainableItems.isEmpty()) {
            return null;
        }
        final double[] a = calculateA(trainableItems, beta, numFactors, c1);
        final double[][] delta = calculateWTWSubset(trainableItems, beta, numFactors, c1 - c0);
        final double[][] b = addInPlace(delta, BTBpR);
        // solve and update factors
        return solve(B, b, A, a);
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateBeta(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        // precomputed matrix
        final double[][] TTTpR = calculateWTWpR(theta, factor, c0, lambdaBeta);
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newBetaVec = calculateNewBetaVector(sample, theta, gamma, gammaBias, betaBias, factor, B, A, TTTpR, c0, c1, globalBias);
            if (newBetaVec != null) {
                setFactorVector(sample.context, beta, newBetaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewBetaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Map<String, double[]> theta,
                                                       @Nonnull final Map<String, double[]> gamma, @Nonnull final Object2DoubleMap<String> gammaBias,
                                                       @Nonnull final Object2DoubleMap<String> betaBias, final int numFactors, @Nonnull final RealMatrix B,
                                                       @Nonnull final RealVector A, @Nonnull final double[][] TTTpR, @Nonnegative final float c0,
                                                       @Nonnegative final float c1, final double globalBias) throws HiveException {
        // filter for trainable users
        final List<Feature> trainableUsers = filterTrainableFeatures(sample.features, theta);
        if (trainableUsers.isEmpty()) {
            return null;
        }
        final List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, gamma);
        final double[] RSD = calculateRSD(sample.context, trainableCooccurringItems, numFactors, betaBias, gammaBias, gamma, globalBias);
        final double[] ApRSD = addInPlace(calculateA(trainableUsers, theta, numFactors, c1), RSD, 1.f);

        final double[][] GTG = calculateWTWSubset(trainableCooccurringItems, gamma, numFactors, 1.f);
        final double[][] delta = calculateWTWSubset(trainableUsers, theta, numFactors, c1 - c0);
        // never add into the precomputed `TTTpR` array, only add into temporary arrays like `delta` and `GTG`
        final double[][] b = addInPlace(addInPlace(delta, GTG), TTTpR);

        // solve and update factors
        return solve(B, b, A, ApRSD);
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateGamma(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newGammaVec = calculateNewGammaVector(sample, beta, gammaBias, betaBias, factor, B, A, lambdaGamma, globalBias);
            if (newGammaVec != null) {
                setFactorVector(sample.context, gamma, newGammaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewGammaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Map<String, double[]> beta,
                                                        @Nonnull final Object2DoubleMap<String> gammaBias, @Nonnull final Object2DoubleMap<String> betaBias,
                                                        @Nonnegative final int numFactors, @Nonnull final RealMatrix B, @Nonnull final RealVector A,
                                                        @Nonnegative final float lambdaGamma, final double globalBias) throws HiveException {
        // filter for trainable items
        final List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }
        final double[][] b = regularize(calculateWTWSubset(trainableCooccurringItems, beta, numFactors, 1.f), lambdaGamma);
        final double[] rsd = calculateRSD(sample.context, trainableCooccurringItems, numFactors, gammaBias, betaBias, beta, globalBias);
        // solve and update factors
        return solve(B, b, A, rsd);
    }

    private static double[][] regularize(@Nonnull final double[][] A, final float lambda) {
        for (int i = 0; i < A.length; i++) {
            A[i][i] += lambda;
        }
        return A;
    }

    private void updateBetaBias(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newBetaBias = calculateNewBias(sample, beta, gamma, gammaBias, globalBias);
            if (newBetaBias != null) {
                setBetaBias(sample.context, newBetaBias);
            }
        }
    }

    public void updateGammaBias(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newGammaBias = calculateNewBias(sample, gamma, beta, betaBias, globalBias);
            if (newGammaBias != null) {
                setGammaBias(sample.context, newGammaBias);
            }
        }
    }

    private void updateGlobalBias(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples) {
        Double newGlobalBias = calculateNewGlobalBias(samples, beta, gamma, betaBias, gammaBias);
        if (newGlobalBias != null) {
            setGlobalBias(newGlobalBias);
        }
    }

    @Nullable
    protected static Double calculateNewGlobalBias(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples, @Nonnull Map<String, double[]> beta,
                                                 @Nonnull Map<String, double[]> gamma, @Nonnull final Object2DoubleMap<String> betaBias,
                                                 @Nonnull final Object2DoubleMap<String> gammaBias) {
        double newGlobalBias = 0.d;
        int numEntriesInSPPMI = 0;
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            // filter for trainable items
            final List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
            if (trainableCooccurringItems.isEmpty()) {
                continue;
            }
            numEntriesInSPPMI += trainableCooccurringItems.size();
            newGlobalBias += calculateGlobalBiasRSD(sample.context, trainableCooccurringItems, beta, gamma, betaBias, gammaBias);
        }
        if (numEntriesInSPPMI == 0) {
            return null;
        }
        return newGlobalBias / numEntriesInSPPMI;
    }

    @VisibleForTesting
    protected static Double calculateNewBias(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Map<String, double[]> beta,
                                             @Nonnull final Map<String, double[]> gamma, @Nonnull final Object2DoubleMap<String> biases,
                                             final double globalBias) {
        // filter for trainable items
        final List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }
        double rsd = calculateBiasRSD(sample.context, trainableCooccurringItems, beta, gamma, biases, globalBias);
        return rsd / trainableCooccurringItems.size();

    }

    @VisibleForTesting
    protected static double calculateGlobalBiasRSD(@Nonnull final String thisItem, @Nonnull final List<Feature> trainableItems,
                                                   @Nonnull final Map<String, double[]> beta, @Nonnull final Map<String, double[]> gamma,
                                                   @Nonnull final Object2DoubleMap<String> betaBias, @Nonnull final Object2DoubleMap<String> gammaBias) {
        double result = 0.d;
        final double[] thisFactorVec = getFactorVector(thisItem, beta);
        final double thisBias = getBias(thisItem, betaBias);
        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            final double[] cooccurVec = getFactorVector(j, gamma);
            double cooccurBias = getBias(j, gammaBias);
            double value = cooccurrence.getValue() - dotProduct(thisFactorVec, cooccurVec) - thisBias - cooccurBias;
            result += value;
        }
        return result;
    }

    @VisibleForTesting
    protected static double calculateBiasRSD(@Nonnull final String thisItem, @Nonnull final List<Feature> trainableItems, @Nonnull final Map<String, double[]> beta,
                                             @Nonnull final Map<String, double[]> gamma, @Nonnull final Object2DoubleMap<String> biases, final double globalBias) {
        double result = 0.d;
        final double[] thisFactorVec = getFactorVector(thisItem, beta);
        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            final double[] cooccurVec = getFactorVector(j, gamma);
            double cooccurBias = getBias(j, biases);
            double value = cooccurrence.getValue() - dotProduct(thisFactorVec, cooccurVec) - cooccurBias - globalBias;
            result += value;
        }
        return result;
    }

    @VisibleForTesting
    @Nonnull
    protected static double[] calculateRSD(@Nonnull final String thisItem, @Nonnull final List<Feature> trainableItems, final int numFactors,
                                           @Nonnull final Object2DoubleMap<String> fixedBias, @Nonnull final Object2DoubleMap<String> changingBias,
                                           @Nonnull final Map<String, double[]> weights, final double globalBias) throws HiveException {

        final double b = getBias(thisItem, fixedBias);
        final double[] accumulator = new double[numFactors];
        for (Feature cooccurrence : trainableItems) {
            final String j = cooccurrence.getFeature();
            double scale = cooccurrence.getValue() - b - getBias(j, changingBias) - globalBias;
            final double[] g = getFactorVector(j, weights);
            addInPlace(accumulator, g, scale);
        }
        return accumulator;
    }

    /**
     * Calculate W' x W plus regularization matrix
     */
    @VisibleForTesting
    @Nonnull
    protected static double[][] calculateWTWpR(@Nonnull final Map<String, double[]> W, @Nonnegative final int numFactors, @Nonnegative final float c0, @Nonnegative final float lambda) {
        double[][] WTW = calculateWTW(W, numFactors, c0);
        return regularize(WTW, lambda);
    }

    private static void checkCondition(final boolean condition, final String errorMessage) throws HiveException {
        if (!condition) {
            throw new HiveException(errorMessage);
        }
    }

    @VisibleForTesting
    @Nonnull
    protected static double[][] addInPlace(@Nonnull final double[][] A, @Nonnull final double[][] B) throws HiveException {
        checkCondition(A.length == A[0].length && A.length == B.length && B.length == B[0].length, ARRAY_NOT_SQUARE_ERR);
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                A[i][j] += B[i][j];
            }
        }
        return A;
    }

    @VisibleForTesting
    @Nonnull
    protected static List<Feature> filterTrainableFeatures(@Nonnull final Feature[] features, @Nonnull final Map<String, double[]> weights) {
        final List<Feature> trainableFeatures = new ArrayList<>();
        String fName;
        for (Feature f : features) {
            fName = f.getFeature();
            if (isTrainable(fName, weights)) {
                trainableFeatures.add(f);
            }
        }
        return trainableFeatures;
    }

    @VisibleForTesting
    protected static RealVector solve(@Nonnull final RealMatrix B, @Nonnull final double[][] dataB, @Nonnull final RealVector A, @Nonnull final double[] dataA) throws HiveException {
        // b * x = a
        // solves for x
        copyData(B, dataB);
        copyData(A, dataA);

        final LUDecomposition LU = new LUDecomposition(B);
        final DecompositionSolver solver = LU.getSolver();

        if (solver.isNonSingular()) {
            return LU.getSolver().solve(A);
        } else {
            SingularValueDecomposition svd = new SingularValueDecomposition(B);
            return svd.getSolver().solve(A);
        }
    }

    private static void copyData(@Nonnull final RealMatrix dst, @Nonnull final double[][] src) throws HiveException {
        checkCondition(dst.getRowDimension() == src.length && dst.getColumnDimension() == src[0].length, DIFFERENT_DIMS_ERR);
        for (int i = 0, rows = dst.getRowDimension(); i < rows; i++) {
            final double[] src_i = src[i];
            for (int j = 0, cols = dst.getColumnDimension(); j < cols; j++) {
                dst.setEntry(i, j, src_i[j]);
            }
        }
    }

    private static void copyData(@Nonnull final RealVector dst, @Nonnull final double[] src) throws HiveException {
        checkCondition(dst.getDimension() == src.length, DIFFERENT_DIMS_ERR);
        for (int i = 0; i < dst.getDimension(); i++) {
            dst.setEntry(i, src[i]);
        }
    }

    private static void copyData(@Nonnull final double[] dst, @Nonnull final RealVector src) throws HiveException {
        checkCondition(dst.length == src.getDimension(), DIFFERENT_DIMS_ERR);
        for (int i = 0; i < dst.length; i++) {
            dst[i] = src.getEntry(i);
        }
    }

    @VisibleForTesting
    @Nonnull
    protected static double[][] calculateWTW(@Nonnull final Map<String, double[]> weights, @Nonnull final int numFactors, @Nonnull final float constant) {
        final double[][] WTW = new double[numFactors][numFactors];
        for (double[] vec : weights.values()) {
            for (int i = 0; i < numFactors; i++) {
                final double[] WTW_f = WTW[i];
                for (int j = 0; j < numFactors; j++) {
                    double val = constant * vec[i] * vec[j];
                    WTW_f[j] += val;
                }
            }
        }
        return WTW;
    }

    @VisibleForTesting
    @Nonnull
    protected static double[][] calculateWTWSubset(@Nonnull final List<Feature> subset, @Nonnull final Map<String, double[]> weights, @Nonnegative final int numFactors, @Nonnegative final float constant) {
        // equivalent to `B_u.T.dot((c1 - c0) * B_u)` in cofacto.py
        final double[][] delta = new double[numFactors][numFactors];
        for (Feature f : subset) {
            final double[] vec = getFactorVector(f.getFeature(), weights);
            for (int i = 0; i < numFactors; i++) {
                final double[] delta_f = delta[i];
                for (int j = 0; j < numFactors; j++) {
                    double val = constant * vec[i] * vec[j];
                    delta_f[j] += val;
                }
            }
        }
        return delta;
    }

    @VisibleForTesting
    @Nonnull
    protected static double[] calculateA(@Nonnull final List<Feature> items, @Nonnull final Map<String, double[]> weights, @Nonnegative final int numFactors, @Nonnegative final float constant) throws HiveException {
        // Equivalent to: a = x_u.dot(c1 * B_u)
        // x_u is a (1, i) matrix of all ones
        // B_u is a (i, F) matrix
        // What it does: sums factor n of each item in B_u
        final double[] A = new double[numFactors];
        for (Feature item : items) {
            double y_ui = item.getValue(); // rating
            addInPlace(A, getFactorVector(item.getFeature(), weights), y_ui);
        }
        for (int a = 0; a < A.length; a++) {
            A[a] *= constant;
        }
        return A;
    }

    @Nullable
    public Double predict(@Nonnull final String user, @Nonnull final String item) {
        if (!theta.containsKey(user) || !beta.containsKey(item)) {
            return null;
        }
        final double[] u = getThetaVector(user), i = getBetaVector(item);
        return dotProduct(u, i);
    }

    @VisibleForTesting
    protected static double dotProduct(@Nonnull final double[] u, @Nonnull final double[] v) {
        double result = 0.d;
        for (int i = 0; i < u.length; i++) {
            result += u[i] * v[i];
        }
        return result;
    }

    public double calculateLoss(@Nonnull final List<CofactorizationUDTF.TrainingSample> users, @Nonnull final List<CofactorizationUDTF.TrainingSample> items) {
        // for speed - can calculate loss on a small subset of the training data
        double mf_loss = calculateMFLoss(users, theta, beta, c0, c1) + calculateMFLoss(items, beta, theta, c0, c1);
        double embed_loss = calculateEmbedLoss(items, beta, gamma, betaBias, gammaBias);
        return mf_loss + embed_loss + sumL2Loss(theta, lambdaTheta) + sumL2Loss(beta, lambdaBeta) + sumL2Loss(gamma, lambdaGamma);

    }

    @VisibleForTesting
    protected static double calculateEmbedLoss(@Nonnull final List<CofactorizationUDTF.TrainingSample> items, @Nonnull final Map<String, double[]> beta,
                                               @Nonnull final Map<String, double[]> gamma, @Nonnull final Object2DoubleMap<String> betaBias,
                                               @Nonnull final Object2DoubleMap<String> gammaBias) {
        double loss = 0.d, val, bBias, gBias;
        double[] bFactors, gFactors;
        String bKey, gKey;
        for (CofactorizationUDTF.TrainingSample item : items) {
            bKey = item.context;
            bFactors = getFactorVector(bKey, beta);
            bBias = getBias(bKey, betaBias);
            for (Feature cooccurrence : item.sppmi) {
                if (!isTrainable(cooccurrence.getFeature(), beta)) {
                    continue;
                }
                gKey = cooccurrence.getFeature();
                gFactors = getFactorVector(gKey, gamma);
                gBias = getBias(gKey, gammaBias);
                val = cooccurrence.getValue() - dotProduct(bFactors, gFactors) - bBias - gBias;
                loss += val * val;
            }
        }
        return loss;
    }

    @VisibleForTesting
    protected static double calculateMFLoss(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples, @Nonnull final Map<String, double[]> contextWeights,
                                            @Nonnull final Map<String, double[]> featureWeights, @Nonnegative final float c0, @Nonnegative final float c1) {
        double loss = 0.d, err, predicted, y;
        double[] contextFactors, ratedFactors;

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            contextFactors = getFactorVector(sample.context, contextWeights);
            // all items / users
            for (double[] unratedFactors : featureWeights.values()) {
                predicted = dotProduct(contextFactors, unratedFactors);
                err = (0.d - predicted);
                loss += c0 * err * err;
            }
            // only rated items / users
            for (Feature f : sample.features) {
                if (!isTrainable(f.getFeature(), featureWeights)) {
                    continue;
                }
                ratedFactors = getFactorVector(f.getFeature(), featureWeights);
                predicted = dotProduct(contextFactors, ratedFactors);
                y = f.getValue();
                err = y - predicted;
                loss += (c1 - c0) * err * err;
            }
        }
        return loss;
    }

    @VisibleForTesting
    protected static double sumL2Loss(@Nonnull final Map<String, double[]> weights, @Nonnegative float lambda) {
        double loss = 0.d;
        for (double[] v : weights.values()) {
            loss += L2Distance(v);
        }
        return lambda * loss;
    }

    @VisibleForTesting
    protected static double L2Distance(@Nonnull final double[] vec) {
        double result = 0.d;
        for (double v : vec) {
            result += v * v;
        }
        return Math.sqrt(result);
    }

    /**
     * Add v to u in-place without creating a new RealVector instance.
     *
     * @param u      array to which v will be added
     * @param v      array containing new values to be added to u
     * @param scalar value to multiply each entry in v before adding to u
     */
    @VisibleForTesting
    @Nonnull
    protected static double[] addInPlace(@Nonnull final double[] u, @Nonnull final double[] v, final double scalar) throws HiveException {
        checkCondition(u.length == v.length, DIFFERENT_DIMS_ERR);
        for (int i = 0; i < u.length; i++) {
            u[i] += scalar * v[i];
        }
        return u;
    }

    private static boolean isTrainable(@Nonnull final String name, @Nonnull final Map<String, double[]> weights) {
        return weights.containsKey(name);
    }

    @Nonnull
    private static Random[] newRandoms(@Nonnegative final int size, final long seed) {
        final Random[] rand = new Random[size];
        for (int i = 0, len = rand.length; i < len; i++) {
            rand[i] = new Random(seed + i);
        }
        return rand;
    }

    private static void uniformFill(@Nonnull final double[] a, @Nonnull final Random rand, final float maxInitValue) {
        for (int i = 0, len = a.length; i < len; i++) {
            double v = rand.nextDouble() * maxInitValue / len;
            a[i] = v;
        }
    }

    private static void gaussianFill(@Nonnull final double[] a, @Nonnull final Random[] rand, @Nonnegative final double stddev) {
        for (int i = 0, len = a.length; i < len; i++) {
            double v = MathUtils.gaussian(0.d, stddev, rand[i]);
            a[i] = v;
        }
    }
}
