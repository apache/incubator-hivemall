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
import hivemall.fm.StringFeature;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.MathUtils;
import it.unimi.dsi.fastutil.objects.Object2DoubleArrayMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import org.apache.commons.math3.linear.*;
import org.apache.hadoop.hive.ql.metadata.HiveException;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

    private static class Prediction implements Comparable<Prediction> {

        private double prediction;
        private int label;
        @Override
        public int compareTo(@Nonnull Prediction other) {
            // descending order
            return -Double.compare(prediction, other.prediction);
        }

    }

    @Nonnegative
    private final int factor;

    // rank matrix initialization
    private final RankInitScheme initScheme;

    private double globalBias;

    // storing trainable latent factors and weights
    private final Weights theta;
    private final Weights beta;
    private final Object2DoubleMap<String> betaBias;
    private final Weights gamma;
    private final Object2DoubleMap<String> gammaBias;

    private final Random[] randU, randI;

    // hyperparameters
    @Nonnegative
    private final float c0, c1;
    private final float lambdaTheta, lambdaBeta, lambdaGamma;

    // validation
    private final CofactorizationUDTF.ValidationMetric validationMetric;
    private final Feature[] validationProbes;
    private final Prediction[] predictions;
    private final int numValPerRecord;
    private String[] users;
    private String[] items;

    // solve
    private final RealMatrix B;
    private final RealVector A;

    // error message strings
    private static final String ARRAY_NOT_SQUARE_ERR = "Array is not square";
    private static final String DIFFERENT_DIMS_ERR = "Matrix, vector or array do not match in size";
    protected static class Weights extends Object2ObjectOpenHashMap<String, double[]> {

        protected Object[] getKey() {
            return key;
        }

        @Nonnull
        String[] getNonnullKeys() {
            final String[] keys = new String[size];
            final Object[] k = (Object[]) key;
            final int len = k.length;
            for (int i = 0, j = 0; i < len; i++) {
                final Object ki = k[i];
                if (ki != null) {
                    keys[j++] = ki.toString();
                }
            }
            return keys;
        }
    }

    public CofactorModel(@Nonnegative final int factor, @Nonnull final RankInitScheme initScheme,
                         @Nonnegative final float c0, @Nonnegative final float c1, @Nonnegative final float lambdaTheta,
                         @Nonnegative final float lambdaBeta, @Nonnegative final float lambdaGamma, final float globalBias,
                         @Nullable CofactorizationUDTF.ValidationMetric validationMetric, @Nonnegative final int numValPerRecord) {

        // rank init scheme is gaussian
        // https://github.com/dawenl/cofactor/blob/master/src/cofacto.py#L98
        this.factor = factor;
        this.initScheme = initScheme;
        this.globalBias = globalBias;
        this.lambdaTheta = lambdaTheta;
        this.lambdaBeta = lambdaBeta;
        this.lambdaGamma = lambdaGamma;

        this.theta = new Weights();
        this.beta = new Weights();
        this.betaBias = new Object2DoubleArrayMap<>();
        this.betaBias.defaultReturnValue(0.d);
        this.gamma = new Weights();
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

        if (validationMetric == null) {
            this.validationMetric = CofactorizationUDTF.ValidationMetric.AUC;
        } else {
            this.validationMetric = validationMetric;
        }

        this.numValPerRecord = numValPerRecord;
        this.validationProbes = new Feature[numValPerRecord];
        this.predictions = new Prediction[numValPerRecord];
        for (int i = 0; i < validationProbes.length; i++) {
            validationProbes[i] = new StringFeature("", 0.d);
            predictions[i] = new Prediction();
        }
    }

    /**
     * Called after UDTF has processed all input records.
     */
    public void finalizeContexts() {
        this.users = theta.getNonnullKeys();
        this.items = beta.getNonnullKeys();
    }

    private void initFactorVector(final String key, final Weights weights) throws HiveException {
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
    private static double[] getFactorVector(String key, Weights weights) {
        return weights.get(key);
    }

    private static void setFactorVector(final String key, final Weights weights, final RealVector factorVector) throws HiveException {
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
    public Weights getTheta() {
        return theta;
    }

    @Nonnull
    public Weights getBeta() {
        return beta;
    }

    @Nonnull
    public Weights getGamma() {
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
    protected static RealVector calculateNewThetaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Weights beta,
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
    protected static RealVector calculateNewBetaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Weights theta,
                                                       @Nonnull final Weights gamma, @Nonnull final Object2DoubleMap<String> gammaBias,
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
    protected static RealVector calculateNewGammaVector(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Weights beta,
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
    protected static Double calculateNewGlobalBias(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples, @Nonnull Weights beta,
                                                   @Nonnull Weights gamma, @Nonnull final Object2DoubleMap<String> betaBias,
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
    protected static Double calculateNewBias(@Nonnull final CofactorizationUDTF.TrainingSample sample, @Nonnull final Weights beta,
                                             @Nonnull final Weights gamma, @Nonnull final Object2DoubleMap<String> biases,
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
                                                   @Nonnull final Weights beta, @Nonnull final Weights gamma,
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
    protected static double calculateBiasRSD(@Nonnull final String thisItem, @Nonnull final List<Feature> trainableItems, @Nonnull final Weights beta,
                                             @Nonnull final Weights gamma, @Nonnull final Object2DoubleMap<String> biases, final double globalBias) {
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
                                           @Nonnull final Weights weights, final double globalBias) throws HiveException {

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
    protected static double[][] calculateWTWpR(@Nonnull final Weights W, @Nonnegative final int numFactors, @Nonnegative final float c0, @Nonnegative final float lambda) {
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
    protected static List<Feature> filterTrainableFeatures(@Nonnull final Feature[] features, @Nonnull final Weights weights) {
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
    protected static double[][] calculateWTW(@Nonnull final Weights weights, @Nonnull final int numFactors, @Nonnull final float constant) {
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
    protected static double[][] calculateWTWSubset(@Nonnull final List<Feature> subset, @Nonnull final Weights weights, @Nonnegative final int numFactors, @Nonnegative final float constant) {
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
    protected static double[] calculateA(@Nonnull final List<Feature> items, @Nonnull final Weights weights, @Nonnegative final int numFactors, @Nonnegative final float constant) throws HiveException {
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
    protected static double calculateEmbedLoss(@Nonnull final List<CofactorizationUDTF.TrainingSample> items, @Nonnull final Weights beta,
                                               @Nonnull final Weights gamma, @Nonnull final Object2DoubleMap<String> betaBias,
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
    protected static double calculateMFLoss(@Nonnull final List<CofactorizationUDTF.TrainingSample> samples, @Nonnull final Weights contextWeights,
                                            @Nonnull final Weights featureWeights, @Nonnegative final float c0, @Nonnegative final float c1) {
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
    protected static double sumL2Loss(@Nonnull final Weights weights, @Nonnegative float lambda) {
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
     * Sample positive and negative validation examples and return a performance metric that
     * should be minimized.
     *
     * @param sample A validation sample
     * @param seed   Integer as seed for random number generator
     * @return Validation metric
     * @throws HiveException
     */
    public Double validate(@Nonnull final CofactorizationUDTF.TrainingSample sample, final int seed) throws HiveException {
        if (!isPredictable(sample.context, sample.isItem())) {
            return null;
        }
        // limit numPos and numNeg
        int numPos = Math.min(sample.features.length, (int) Math.ceil(this.numValPerRecord * 0.5));
        int numNeg = Math.min(this.numValPerRecord - numPos, sample.isItem() ? users.length : items.length);

        getValidationExamples(numPos, numNeg, sample.features, sample.isItem(), validationProbes, seed);
        if (validationMetric == CofactorizationUDTF.ValidationMetric.AUC) {
            return -calculateAUC(validationProbes, predictions, sample, numPos, numNeg);
        } else {
            return calculateLoss(validationProbes, sample, numPos, numNeg);
        }
    }

    private boolean isPredictable(@Nonnull final String context, final boolean isItem) {
        if (isItem) {
            return beta.containsKey(context);
        } else {
            return theta.containsKey(context);
        }
    }

    /**
     * TODO: not implemented
     *
     * @return
     */
    private double calculateLoss(Feature[] validationProbes, CofactorizationUDTF.TrainingSample sample, int numPos, int numNeg) {
        return 0d;
    }

    /**
     * Calculates area under curve for validation metric.
     */
    private double calculateAUC(@Nonnull final Feature[] validationProbes, @Nonnull final Prediction[] predictions, CofactorizationUDTF.TrainingSample sample, final int numPos, final int numNeg) {
        // make predictions for positive and then negative examples
        int nextIdx = fillPredictions(validationProbes, predictions, sample, 0, numPos, 0, 1);
        int endIdx = fillPredictions(validationProbes, predictions, sample, nextIdx, numPos + numNeg, nextIdx, 0);

        // sort in descending order for all filled predictions
        Arrays.sort(predictions, 0, endIdx);

        double area = 0d, scorePrev = Double.MIN_VALUE;
        int fp = 0, tp = 0;
        int fpPrev = 0, tpPrev = 0;

        for (int i = 0; i < endIdx; i++) {
            final Prediction p = predictions[i];
            if (p.prediction != scorePrev) {
                area += trapezoid(fp, fpPrev, tp, tpPrev);
                scorePrev = p.prediction;
                fpPrev = fp;
                tpPrev = tp;
            }
            if (p.label == 1) {
                tp += 1;
            } else {
                fp += 1;
            }
        }
        area += trapezoid(fp, fpPrev, tp, tpPrev);
        return area / (tp * fp);
    }

    /**
     * Calculates area of a trapezoid.
     */
    private static double trapezoid(final int x1, final int x2, final int y1, final int y2) {
        final int base = Math.abs(x1 - x2);
        final double height = (y1 + y2) * 0.5;
        return base * height;
    }

    /**
     * Fill an array of predictions.
     * @return index of the next empty entry in {@code predictions} array
     */
    private int fillPredictions(@Nonnull final Feature[] validationProbes, @Nonnull final Prediction[] predictions, @Nonnull final CofactorizationUDTF.TrainingSample sample,
                                final int lo, final int hi, int fillIdx, final int label) {
        for (int i = lo; i < hi; i++) {
            final Feature pos = validationProbes[i];
            final Double pred;
            if (sample.isItem()) {
                pred = predict(pos.getFeature(), sample.context);
            } else {
                pred = predict(sample.context, pos.getFeature());
            }
            if (pred == null) {
                continue;
            }
            predictions[fillIdx].prediction = pred;
            predictions[fillIdx].label = label;
            fillIdx++;
        }
        return fillIdx;
    }

    /**
     * Sample positive and negative samples.
     * @return number of negatives that were successfully sampled
     */
    private void getValidationExamples(final int numPos, final int numNeg, @Nonnull final Feature[] positives, final boolean isContextAnItem,
                                       @Nonnull final Feature[] validationProbes, final int seed) {
        final Random rand = new Random(seed);
        samplePositives(numPos, positives, validationProbes, rand);
        final String[] keys = isContextAnItem ? users : items;
        sampleNegatives(numPos, numNeg, validationProbes, keys, rand);
    }

    /**
     * Samples negative examples.
     */
    @VisibleForTesting
    protected static void sampleNegatives(final int numPos, final int numNeg, @Nonnull final Feature[] validationProbes,
                                          @Nonnull final String[] keys, @Nonnull final Random rand) {
        // sample numPos positive examples without replacement
        for (int i = numPos, size = numPos + numNeg; i < size; i++) {
            final String negKey = keys[rand.nextInt(keys.length)];
            validationProbes[i].setFeature(negKey);
            validationProbes[i].setValue(0.d);
        }
    }

    private static void samplePositives(final int numPos, @Nonnull final Feature[] positives, @Nonnull final Feature[] validationProbes, @Nonnull final Random rand) {
        // sample numPos positive examples without replacement
        for (int i = 0; i < numPos; i++) {
            validationProbes[i] = positives[rand.nextInt(positives.length)];
        }
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

    private static boolean isTrainable(@Nonnull final String name, @Nonnull final Weights weights) {
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
