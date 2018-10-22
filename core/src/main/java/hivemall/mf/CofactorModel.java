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

    @Nonnull
    private double globalBias;

    // storing trainable latent factors and weights
    private final Map<String, double[]> theta;
    private final Map<String, double[]> beta;
    private final Object2DoubleMap<String> betaBias;
    private final Map<String, double[]> gamma;
    private final Object2DoubleMap<String> gammaBias;

    private final Random[] randU, randI;

    // hyperparameters
    private final float c0, c1;
    private final float lambdaTheta, lambdaBeta, lambdaGamma;

    // solve
    private final RealMatrix B;
    private final RealVector A;

    // error message strings
    private static final String ARRAY_NOT_SQUARE_ERR = "Array is not square";
    private static final String DIFFERENT_DIMS_ERR = "Matrix, vector or array do not match in size";

    public CofactorModel(@Nonnegative int factor, @Nonnull RankInitScheme initScheme,
                         float c0, float c1, float lambdaTheta, float lambdaBeta, float lambdaGamma) {

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

    private void initFactorVector(final String key, final Map<String, double[]> weights) {
        if (weights.containsKey(key)) {
            return;
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

    private static double[] getFactorVector(String key, Map<String, double[]> weights) {
        return weights.get(key);
    }

    private static void setFactorVector(String key, Map<String, double[]> weights, RealVector factorVector) throws HiveException {
        double[] vec = weights.get(key);
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

    public void recordContext(String context, Boolean isItem) {
        if (isItem) {
            initFactorVector(context, beta);
            initFactorVector(context, gamma);
        } else {
            initFactorVector(context, theta);
        }
    }

    public double[] getGammaVector(final String key) {
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

    public double[] getThetaVector(final String key) {
        return getFactorVector(key, theta);
    }

    public double[] getBetaVector(final String key) {
        return getFactorVector(key, beta);
    }

    public double getBetaBias(final String key) {
        return getBias(key, betaBias);
    }

    public void setBetaBias(final String key, final double value) {
        setBias(key, betaBias, value);
    }

    public Map<String, double[]> getTheta() {
        return theta;
    }

    public Map<String, double[]> getBeta() {
        return beta;
    }

    public Map<String, double[]> getGamma() {
        return gamma;
    }

    public Object2DoubleMap<String> getBetaBiases() {
        return betaBias;
    }

    public Object2DoubleMap<String> getGammaBiases() {
        return gammaBias;
    }

    public void updateWithUsers(List<CofactorizationUDTF.TrainingSample> users) throws HiveException {
        updateTheta(users);
    }

    public void updateWithItems(List<CofactorizationUDTF.TrainingSample> items) throws HiveException {
        updateBeta(items);
        updateGamma(items);
        updateBetaBias(items);
        updateGammaBias(items);
    }

    /**
     * Update latent factors of the users in the provided mini-batch.
     */
    private void updateTheta(List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        // initialize item factors
        // items should only be trainable if the dataset contains a major entry for that item (which it may not)
        // variable names follow cofacto.py
        double[][] BTBpR = calculateWTWpR(beta, factor, c0, lambdaTheta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newThetaVec = calculateNewThetaVector(sample, beta, factor, B, A, BTBpR, c0, c1);
            if (newThetaVec != null) {
                setFactorVector(sample.context, theta, newThetaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewThetaVector(CofactorizationUDTF.TrainingSample sample, Map<String, double[]> beta,
                                                        int numFactors, RealMatrix B, RealVector A, double[][] BTBpR, float c0, float c1) throws HiveException {
        // filter for trainable items
        List<Feature> trainableItems = filterTrainableFeatures(sample.features, beta);
        // TODO: is this correct behaviour?
        if (trainableItems.isEmpty()) {
            return null;
        }

        double[] a = calculateA(trainableItems, beta, numFactors, c1);

        double[][] delta = calculateWTWSubset(trainableItems, beta, numFactors, c1 - c0);
        double[][] b = addInPlace(delta, BTBpR);

        // solve and update factors
        return solve(B, b, A, a);
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateBeta(List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        // precomputed matrix
        double[][] TTTpR = calculateWTWpR(theta, factor, c0, lambdaBeta);

        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newBetaVec = calculateNewBetaVector(sample, theta, gamma, gammaBias, betaBias, factor, B, A, TTTpR, c0, c1);
            if (newBetaVec != null) {
                setFactorVector(sample.context, beta, newBetaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewBetaVector(CofactorizationUDTF.TrainingSample sample, Map<String, double[]> theta,
                                                       Map<String, double[]> gamma, Object2DoubleMap<String> gammaBias,
                                                       Object2DoubleMap<String> betaBias, int numFactors, RealMatrix B, RealVector A,
                                                       double[][] TTTpR, float c0, float c1) throws HiveException {
        // filter for trainable users
        List<Feature> trainableUsers = filterTrainableFeatures(sample.features, theta);
        // TODO: is this correct behaviour?
        if (trainableUsers.isEmpty()) {
            return null;
        }

        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, gamma);
        double[] RSD = calculateRSD(sample.context, trainableCooccurringItems, numFactors, betaBias, gammaBias, gamma);
        double[] ApRSD = addInPlace(calculateA(trainableUsers, theta, numFactors, c1), RSD, 1.f);

        double[][] GTG = calculateWTWSubset(trainableCooccurringItems, gamma, numFactors, 1.f);
        double[][] delta = calculateWTWSubset(trainableUsers, theta, numFactors, c1 - c0);
        // never add into the precomputed `TTTpR` array, only add into temporary arrays like `delta` and `GTG`
        double[][] b = addInPlace(addInPlace(delta, GTG), TTTpR);

        // solve and update factors
        return solve(B, b, A, ApRSD);
    }

    /**
     * Update latent factors of the items in the provided mini-batch.
     */
    private void updateGamma(List<CofactorizationUDTF.TrainingSample> samples) throws HiveException {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            RealVector newGammaVec = calculateNewGammaVector(sample, beta, gammaBias, betaBias, factor, B, A, lambdaGamma);
            if (newGammaVec != null) {
                setFactorVector(sample.context, gamma, newGammaVec);
            }
        }
    }

    @VisibleForTesting
    protected static RealVector calculateNewGammaVector(CofactorizationUDTF.TrainingSample sample, Map<String, double[]> beta,
                                                        Object2DoubleMap<String> gammaBias, Object2DoubleMap<String> betaBias,
                                                        int numFactors, RealMatrix B, RealVector A, float lambdaGamma) throws HiveException {
        // filter for trainable items
        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        // TODO: is this correct behaviour?
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }

        double[][] b = regularize(calculateWTWSubset(trainableCooccurringItems, beta, numFactors, 1.f), lambdaGamma);
        double[] rsd = calculateRSD(sample.context, trainableCooccurringItems, numFactors, gammaBias, betaBias, beta);

        // solve and update factors
        return solve(B, b, A, rsd);
    }

    private static double[][] regularize(double[][] A, float lambda) {
        for (int i = 0; i < A.length; i++) {
            A[i][i] += lambda;
        }
        return A;
    }

    private void updateBetaBias(List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newBetaBias = calculateNewBias(sample, beta, gamma, gammaBias);
            // TODO: is this correct behaviour?
            if (newBetaBias != null) {
                setBetaBias(sample.context, newBetaBias);
            }
        }
    }

    public void updateGammaBias(List<CofactorizationUDTF.TrainingSample> samples) {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            Double newGammaBias = calculateNewBias(sample, gamma, beta, betaBias);
            // TODO: is this correct behaviour?
            if (newGammaBias != null) {
                setGammaBias(sample.context, newGammaBias);
            }
        }
    }

    @VisibleForTesting
    protected static Double calculateNewBias(CofactorizationUDTF.TrainingSample sample, Map<String, double[]> beta,
                                             Map<String, double[]> gamma, Object2DoubleMap<String> biases) {
        // filter for trainable items
        List<Feature> trainableCooccurringItems = filterTrainableFeatures(sample.sppmi, beta);
        if (trainableCooccurringItems.isEmpty()) {
            return null;
        }

        double rsd = calculateBiasRSD(sample.context, trainableCooccurringItems, beta, gamma, biases);
        return rsd / trainableCooccurringItems.size();

    }

    @VisibleForTesting
    protected static double calculateBiasRSD(String thisItem, List<Feature> trainableItems, Map<String, double[]> beta,
                                             Map<String, double[]> gamma, Object2DoubleMap<String> biases) {
        double result = 0.d, cooccurBias;
        double[] thisFactorVec = getFactorVector(thisItem, beta);
        double[] cooccurVec;

        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            cooccurVec = getFactorVector(j, gamma);
            cooccurBias = getBias(j, biases);
            double value = cooccurrence.getValue() - dotProduct(thisFactorVec, cooccurVec) - cooccurBias;
            result += value;
        }
        return result;
    }

    @VisibleForTesting
    protected static double[] calculateRSD(String thisItem, List<Feature> trainableItems, int numFactors,
                                           Object2DoubleMap<String> fixedBias, Object2DoubleMap<String> changingBias,
                                           Map<String, double[]> weights) throws HiveException {

        double b = getBias(thisItem, fixedBias);

        double[] accumulator = new double[numFactors];

        // m_ij is named the same as in cofacto.py
        for (Feature cooccurrence : trainableItems) {
            String j = cooccurrence.getFeature();
            double scale = cooccurrence.getValue() - b - getBias(j, changingBias);
            double[] g = getFactorVector(j, weights);
            addInPlace(accumulator, g, scale);
        }
        return accumulator;
    }

    /**
     * Calculate W' x W plus regularization matrix
     */
    @VisibleForTesting
    protected static double[][] calculateWTWpR(Map<String, double[]> W, int numFactors, float c0, float lambda) {
        double[][] WTW = calculateWTW(W, numFactors, c0);
        return regularize(WTW, lambda);
    }

    private static void checkCondition(boolean condition, String errorMessage) throws HiveException {
        if (!condition) {
            throw new HiveException(errorMessage);
        }
    }

    @VisibleForTesting
    protected static double[][] addInPlace(@Nonnull double[][] A, @Nonnull double[][] B) throws HiveException {
        checkCondition(A.length == A[0].length && A.length == B.length && B.length == B[0].length, ARRAY_NOT_SQUARE_ERR);
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                A[i][j] += B[i][j];
            }
        }
        return A;
    }

    @VisibleForTesting
    protected static List<Feature> filterTrainableFeatures(Feature[] features, Map<String, double[]> weights) {
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

    @VisibleForTesting
    protected static RealVector solve(final RealMatrix B, final double[][] dataB, final RealVector A, final double[] dataA) throws HiveException {
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

    private static void copyData(final RealMatrix dst, final double[][] src) throws HiveException {
        checkCondition(dst.getRowDimension() == src.length && dst.getColumnDimension() == src[0].length, DIFFERENT_DIMS_ERR);
        for (int i = 0, rows = dst.getRowDimension(); i < rows; i++) {
            final double[] src_i = src[i];
            for (int j = 0, cols = dst.getColumnDimension(); j < cols; j++) {
                dst.setEntry(i, j, src_i[j]);
            }
        }
    }

    private static void copyData(final RealVector dst, final double[] src) throws HiveException {
        checkCondition(dst.getDimension() == src.length, DIFFERENT_DIMS_ERR);
        for (int i = 0; i < dst.getDimension(); i++) {
            dst.setEntry(i, src[i]);
        }
    }

    private static void copyData(final double[] dst, final RealVector src) throws HiveException {
        checkCondition(dst.length == src.getDimension(), DIFFERENT_DIMS_ERR);
        for (int i = 0; i < dst.length; i++) {
            dst[i] = src.getEntry(i);
        }
    }

    @VisibleForTesting
    protected static double[][] calculateWTW(final Map<String, double[]> weights, final int numFactors, final float constant) {
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
    protected static double[][] calculateWTWSubset(List<Feature> subset, Map<String, double[]> weights, int numFactors, float constant) {
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
    protected static double[] calculateA(List<Feature> items, Map<String, double[]> weights, int numFactors, float constant) throws HiveException {
        // Equivalent to: a = x_u.dot(c1 * B_u)
        // x_u is a (1, i) matrix of all ones
        // B_u is a (i, F) matrix
        // What it does: sums factor n of each item in B_u
//        clearArray(A);
        double[] A = new double[numFactors];
        for (Feature item : items) {
            double y_ui = item.getValue(); // rating
            addInPlace(A, getFactorVector(item.getFeature(), weights), y_ui);
        }
        for (int a = 0; a < A.length; a++) {
            A[a] *= constant;
        }
        return A;
    }

    private static void clearArray(double[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = 0.d;
        }
    }

    public Double predict(String user, String item) {
        if (!isTrainable(user, theta) || !isTrainable(item, beta)) {
            return null;
        }
        double[] u = getThetaVector(user), i = getBetaVector(item);
        return dotProduct(u, i);
    }

    @VisibleForTesting
    protected static double dotProduct(double[] u, double[] v) {
        double result = 0.d;
        for (int i = 0; i < u.length; i++) {
            result += u[i] * v[i];
        }
        return result;
    }

    public Double calculateLoss(List<CofactorizationUDTF.TrainingSample> users, List<CofactorizationUDTF.TrainingSample> items) {
        // for speed - can calculate loss on a small subset of the training data
        double mf_loss = calculateMFLoss(users, theta, beta, c0, c1) + calculateMFLoss(items, beta, theta, c0, c1);
        double embed_loss = calculateEmbedLoss(items, beta, gamma, betaBias, gammaBias);
        return mf_loss + embed_loss + sumL2Loss(theta, lambdaTheta) + sumL2Loss(beta, lambdaBeta) + sumL2Loss(gamma, lambdaGamma);

    }

    @VisibleForTesting
    protected static double calculateEmbedLoss(List<CofactorizationUDTF.TrainingSample> items, Map<String, double[]> beta,
                                               Map<String, double[]> gamma, Object2DoubleMap<String> betaBias,
                                               Object2DoubleMap<String> gammaBias) {
        double loss = 0.d, val, bBias, gBias;
        double[] bFactors, gFactors;
        String bKey, gKey;
        for (CofactorizationUDTF.TrainingSample item: items) {
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
    protected static double calculateMFLoss(List<CofactorizationUDTF.TrainingSample> samples, Map<String, double[]> contextWeights,
                                            Map<String, double[]> featureWeights, float c0, float c1) {
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
    protected static double sumL2Loss(Map<String, double[]> weights, float lambda) {
        double loss = 0.d;
        for (double[] v : weights.values()) {
            loss += L2Distance(v);
        }
        return lambda * loss;
    }

    @VisibleForTesting
    protected static double L2Distance(double[] vec) {
        double result = 0.d;
        for (double v : vec) {
            result +=  v * v;
        }
        return Math.sqrt(result);
    }

    /**
     * Add v to u in-place without creating a new RealVector instance.
     * @param u array to which v will be added
     * @param v array containing new values to be added to u
     * @param scalar value to multiply each entry in v before adding to u
     */
    @VisibleForTesting
    protected static double[] addInPlace(double[] u, double[] v, double scalar) throws HiveException {
        checkCondition(u.length == v.length, DIFFERENT_DIMS_ERR);
        for (int i = 0; i < u.length; i++) {
            u[i] += scalar * v[i];
        }
        return u;
    }

    private static boolean isTrainable(String name, Map<String, double[]> weights) {
        return weights.containsKey(name);
    }

    @Nonnull
    private static Random[] newRandoms(final int size, final long seed) {
        final Random[] rand = new Random[size];
        for (int i = 0, len = rand.length; i < len; i++) {
            rand[i] = new Random(seed + i);
        }
        return rand;
    }

    private static void uniformFill(final double[] a, final Random rand, final float maxInitValue) {
        for (int i = 0, len = a.length; i < len; i++) {
            double v = rand.nextDouble() * maxInitValue / len;
            a[i] = v;
        }
    }

    private static void gaussianFill(final double[] a, final Random[] rand, final double stddev) {
        for (int i = 0, len = a.length; i < len; i++) {
            double v = MathUtils.gaussian(0.d, stddev, rand[i]);
            a[i] = v;
        }
    }
}
