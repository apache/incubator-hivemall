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
import hivemall.fm.StringFeature;
import it.unimi.dsi.fastutil.objects.Object2DoubleArrayMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Test;

import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class CofactorModelTest {
    private static final double EPSILON = 1e-3;
    private static final int NUM_FACTORS = 2;

    // items
    private static final String TOOTHBRUSH = "toothbrush";
    private static final String TOOTHPASTE = "toothpaste";
    private static final String SHAVER = "shaver";

    // users
    private static final String MAKOTO = "makoto";
    private static final String TAKUYA = "takuya";
    private static final String JACKSON = "jackson";
    private static final String ALIEN = "alien";

    @Test
    public void calculateWTW() {
        CofactorModel.Weights weights = getTestBeta();

        double[][] expectedWTW = new double[][]{
                {0.63, -0.238},
                {-0.238, 0.346}
        };

        double[][] actualWTW = CofactorModel.calculateWTW(weights, 2, 0.1f);
        Assert.assertTrue(matricesAreEqual(actualWTW, expectedWTW));
    }

    @Test
    public void calculateA() throws HiveException {
        CofactorModel.Weights weights = getTestBeta();
        List<Feature> items = getSubset_itemFeatureList_explicitFeedback();
        double[] actual = CofactorModel.calculateA(items, weights, NUM_FACTORS, 0.5f);
        double[] expected = new double[]{-2.05, 3.15};
        Assert.assertArrayEquals(actual, expected, EPSILON);
    }

    @Test
    public void calculateWTWSubset() throws HiveException {
        CofactorModel.Weights weights = getTestBeta();
        List<Feature> items = getSubset_itemFeatureList_explicitFeedback();

        double[][] actual = CofactorModel.calculateWTWSubset(items, weights, NUM_FACTORS, 0.9f);
        double[][] expected = new double[][]{
                {4.581, -3.033},
                {-3.033, 2.385}
        };

        Assert.assertTrue(matricesAreEqual(actual, expected));
    }

    @Test
    public void calculateNewThetaVector() throws HiveException {
        final float c0 = 0.1f, c1 = 1.f, lambdaTheta = 1e-5f;
        CofactorModel.Weights beta = getTestBeta();

        double[][] BTBpR = CofactorModel.calculateWTWpR(beta, NUM_FACTORS, c0, lambdaTheta);
        double[][] initialBTBpR = copyArray(BTBpR);

        RealMatrix B = new Array2DRowRealMatrix(NUM_FACTORS, NUM_FACTORS);
        RealVector A = new ArrayRealVector(NUM_FACTORS);

        CofactorizationUDTF.TrainingSample currentUser = new CofactorizationUDTF.TrainingSample(
                JACKSON, getSubset_itemFeatureVector_implicitFeedback(), false, null);

        RealVector actual = CofactorModel.calculateNewThetaVector(currentUser, beta, NUM_FACTORS, B, A, BTBpR, c0, c1);
        Assert.assertNotNull(actual);

        // ensure that TTTpR has not been accidentally changed after one update
        Assert.assertTrue(matricesAreEqual(initialBTBpR, BTBpR));

        double[] expected = new double[]{0.44514062, 1.22886953};
        Assert.assertArrayEquals(expected, actual.toArray(), EPSILON);
    }

    @Test
    public void calculateRSD() throws HiveException {
        double[] actual = CofactorModel.calculateRSD(
                TOOTHBRUSH,
                getToothbrushSPPMIList(),
                NUM_FACTORS,
                getTestBetaBias(),
                getTestGammaBias(),
                getTestGamma(),
                0.d);
        double[] expected = new double[]{2.656, 0.154};
        Assert.assertArrayEquals(expected, actual, EPSILON);
    }

    @Test
    public void calculateNewBetaVector() throws HiveException {
        final float c0 = 0.1f, c1 = 1.f, lambdaBeta = 1e-5f;

        Object2DoubleMap<String> betaBias = getTestBetaBias();
        Object2DoubleMap<String> gammaBias = getTestGammaBias();
        CofactorModel.Weights gamma = getTestGamma();
        CofactorModel.Weights theta = getTestTheta();

        RealMatrix B = new Array2DRowRealMatrix(NUM_FACTORS, NUM_FACTORS);
        RealVector A = new ArrayRealVector(NUM_FACTORS);

        // solve for new weights for toothbrush
        CofactorizationUDTF.TrainingSample toothbrush = getItemSamples(true).get(0);

        double[][] TTTpR = CofactorModel.calculateWTWpR(theta, NUM_FACTORS, c0, lambdaBeta);
        double[][] initialTTTpR = copyArray(TTTpR);

        // zero bias: solve and update factors
        RealVector actual = CofactorModel.calculateNewBetaVector(toothbrush, theta, gamma, gammaBias, betaBias, NUM_FACTORS, B, A, TTTpR, c0, c1, 0.d);
        Assert.assertNotNull(actual);
        // ensure that TTTpR has not been accidentally changed after one update
        Assert.assertTrue(matricesAreEqual(initialTTTpR, TTTpR));
        double[] expected = new double[]{0.07613607, -2.98883404};
        Assert.assertArrayEquals(expected, actual.toArray(), EPSILON);

        // non-zero bias: solve and update factors
        actual = CofactorModel.calculateNewBetaVector(toothbrush, theta, gamma, gammaBias, betaBias, NUM_FACTORS, B, A, TTTpR, c0, c1, 2.5d);
        Assert.assertNotNull(actual);
        // ensure that TTTpR has not been accidentally changed after one update
        Assert.assertTrue(matricesAreEqual(initialTTTpR, TTTpR));
        expected = new double[]{-0.92773117, -4.03186978};
        Assert.assertArrayEquals(expected, actual.toArray(), EPSILON);
    }

    @Test
    public void calculateNewGlobalBias() {
        CofactorModel.Weights beta = getTestBeta();
        CofactorModel.Weights gamma = getTestGamma();
        Object2DoubleMap<String> betaBias = getTestBetaBias();
        Object2DoubleMap<String> gammaBias = getTestGammaBias();

        List<CofactorizationUDTF.TrainingSample> items = getItemSamples(false);
        Double actual = CofactorModel.calculateNewGlobalBias(items, beta, gamma, betaBias, gammaBias);
        Assert.assertNotNull(actual);
        Assert.assertEquals(-0.2667, actual, EPSILON);
    }

    private static double[][] copyArray(double[][] A) {
        double[][] newA = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                newA[i][j] = A[i][j];
            }
        }
        return newA;
    }

    @Test
    public void calculateNewGammaVector() throws HiveException {
        final float lambdaGamma = 1e-5f;

        Object2DoubleMap<String> betaBias = getTestBetaBias();
        Object2DoubleMap<String> gammaBias = getTestGammaBias();
        CofactorModel.Weights beta = getTestBeta();

        RealMatrix B = new Array2DRowRealMatrix(NUM_FACTORS, NUM_FACTORS);
        RealVector A = new ArrayRealVector(NUM_FACTORS);

        CofactorizationUDTF.TrainingSample currentItem = getItemSamples(false).get(0);

        // zero global bias
        RealVector actual = CofactorModel.calculateNewGammaVector(currentItem, beta, gammaBias, betaBias, NUM_FACTORS, B, A, lambdaGamma, 0.d);
        Assert.assertNotNull(actual);
        double[] expected = new double[]{0.95828914, -1.48234826};
        Assert.assertArrayEquals(expected, actual.toArray(), EPSILON);

        // non-zero global bias
        actual = CofactorModel.calculateNewGammaVector(currentItem, beta, gammaBias, betaBias, NUM_FACTORS, B, A, lambdaGamma, 2.5d);
        Assert.assertNotNull(actual);
        expected = new double[]{0.49037982, -3.68822023};
        Assert.assertArrayEquals(expected, actual.toArray(), EPSILON);
    }

    @Test
    public void calculateNewBias_forBetaBias_returnsNonNull() throws HiveException {
        Object2DoubleMap<String> gammaBias = getTestGammaBias();
        CofactorModel.Weights beta = getTestBeta();
        CofactorModel.Weights gamma = getTestGamma();

        CofactorizationUDTF.TrainingSample currentItem = getItemSamples(false).get(0);

        // zero global bias
        Double actual = CofactorModel.calculateNewBias(currentItem, beta, gamma, gammaBias, 0.d);
        Assert.assertNotNull(actual);
        Assert.assertEquals(-0.235, actual, EPSILON);

        // non-zero global bias
        actual = CofactorModel.calculateNewBias(currentItem, beta, gamma, gammaBias, 2.5d);
        Assert.assertNotNull(actual);
        Assert.assertEquals(-2.735, actual, EPSILON);
    }

    @Test
    public void recordAsParent() throws HiveException {
        CofactorModel model = new CofactorModel(NUM_FACTORS, CofactorModel.RankInitScheme.gaussian, 0.1f, 1.f, 1e-5f, 1e-5f, 1.f, 0.f);

        Assert.assertNull(model.getThetaVector(JACKSON));
        model.recordContext(JACKSON, false);
        Assert.assertNotNull(model.getThetaVector(JACKSON));

        Assert.assertNull(model.getBetaVector(TOOTHBRUSH));
        Assert.assertNull(model.getGammaVector(TOOTHBRUSH));
        model.recordContext(TOOTHBRUSH, true);
        Assert.assertNotNull(model.getBetaVector(TOOTHBRUSH));
        Assert.assertNotNull(model.getGammaVector(TOOTHBRUSH));
    }

    @Test
    public void L2Distance() throws HiveException {
        double[] v = new double[]{0.1, 2.3, 5.3};
        double actual = CofactorModel.L2Distance(v);
        double expected = 5.7784d;
        Assert.assertEquals(actual, expected, EPSILON);
    }

    @Test
    public void calculateMFLoss_allFeaturesAreTrainable() throws HiveException {
        List<CofactorizationUDTF.TrainingSample> samples = getSamples_itemAsContext_allUsersInTheta();
        CofactorModel.Weights beta = getTestBeta();
        CofactorModel.Weights theta = getTestTheta();
        double actual = CofactorModel.calculateMFLoss(samples, beta, theta, 0.1f, 1.f);
        double expected = 0.7157;
        Assert.assertEquals(actual, expected, EPSILON);
    }

    @Test
    public void calculateMFLoss_oneFeatureNotTrainable() throws HiveException {
        // tests case where a user found in the item's feature array
        // was not also distributed to the same UDTF instance
        List<CofactorizationUDTF.TrainingSample> samples = getSamples_itemAsContext_oneUserNotInTheta();
        CofactorModel.Weights beta = getTestBeta();
        CofactorModel.Weights theta = getTestTheta();
        double actual = CofactorModel.calculateMFLoss(samples, beta, theta, 0.1f, 1.f);
        double expected = 0.7157;
        Assert.assertEquals(actual, expected, EPSILON);
    }

    @Test
    public void calculateEmbedLoss() {
        List<CofactorizationUDTF.TrainingSample> samples = getSamples_itemAsContext_allUsersInTheta();
        CofactorModel.Weights beta = getTestBeta();
        CofactorModel.Weights gamma = getTestGamma();
        Object2DoubleMap<String> betaBias = getTestBetaBias();
        Object2DoubleMap<String> gammaBias = getTestGammaBias();

        double actual = CofactorModel.calculateEmbedLoss(samples, beta, gamma, betaBias, gammaBias);
        double expected = 2.756;
        Assert.assertEquals(expected, actual, EPSILON);
    }

    @Test
    public void dotProduct() {
        double[] u = new double[]{0.1, 5.1, 3.2};
        double[] v = new double[]{1, 2, 3};
        Assert.assertEquals(CofactorModel.dotProduct(u, v), 19.9, EPSILON);
    }

    @Test
    public void addInPlaceArray1D() throws HiveException {
        double[] u = new double[]{0.1, 5.1, 3.2};
        double[] v = new double[]{1, 2, 3};

        double[] actual = CofactorModel.addInPlace(u, v, 1.f);
        double[] expected = new double[]{1.1, 7.1, 6.2};
        Assert.assertArrayEquals(u, expected, EPSILON);
        Assert.assertArrayEquals(actual, expected, EPSILON);
    }

    @Test
    public void addInPlaceArray2D() throws HiveException {
        double[][] u = new double[][]{{0.1, 5.1}, {3.2, 1.2}};
        double[][] v = new double[][]{{1, 2}, {3, 4}};

        double[][] actual = CofactorModel.addInPlace(u, v);
        double[][] expected = new double[][]{{1.1, 7.1}, {6.2, 5.2}};
        Assert.assertTrue(matricesAreEqual(u, expected));
        Assert.assertTrue(matricesAreEqual(actual, expected));
    }

    @Test
    public void smallTrainingTest_implicitFeedback() throws HiveException {
        final boolean IS_FEEDBACK_EXPLICIT = false;
        CofactorModel.RankInitScheme init = CofactorModel.RankInitScheme.gaussian;
        init.setInitStdDev(1.0f);

        CofactorModel model = new CofactorModel(NUM_FACTORS, init,
                0.1f, 1.f, 1e-5f, 1e-5f, 1.f, 0.f);
        int iterations = 5;
        List<CofactorizationUDTF.TrainingSample> users = getUserSamples(IS_FEEDBACK_EXPLICIT);
        List<CofactorizationUDTF.TrainingSample> items = getItemSamples(IS_FEEDBACK_EXPLICIT);

        // record features
        recordContexts(model, users, false);
        recordContexts(model, items, true);

        double prevLoss = Double.MAX_VALUE;
        for (int i = 0; i < iterations; i++) {
            model.updateWithUsers(users);
            model.updateWithItems(items);
            Double loss = model.calculateLoss(users, items);
            Assert.assertNotNull(loss);
            Assert.assertTrue(loss < prevLoss);
            prevLoss = loss;
        }

        // assert that the user-item predictions after N iterations is identical to expected predictions
//        String expected = "makoto -> (toothpaste:0.976), (toothbrush:0.942), (shaver:1.076), \n" +
//                "takuya -> (toothpaste:1.001), (toothbrush:-0.167), (shaver:0.173), \n" +
//                "jackson -> (toothpaste:1.031), (toothbrush:0.715), (shaver:0.906), \n";
        String predictionString = generatePredictionString(model, users, items);
        System.out.println(predictionString);
//        Assert.assertEquals(predictionString, expected);
    }

    @Test
    public void smallTrainingTest_explicitFeedback() throws HiveException {
        final boolean IS_FEEDBACK_EXPLICIT = true;
        CofactorModel.RankInitScheme init = CofactorModel.RankInitScheme.gaussian;
        init.setInitStdDev(1.0f);

        CofactorModel model = new CofactorModel(NUM_FACTORS, init,
                0.1f, 1.f, 1e-5f, 1e-5f, 1.f, 0.f);
        int iterations = 5;
        List<CofactorizationUDTF.TrainingSample> users = getUserSamples(IS_FEEDBACK_EXPLICIT);
        List<CofactorizationUDTF.TrainingSample> items = getItemSamples(IS_FEEDBACK_EXPLICIT);

        // record features
        recordContexts(model, users, false);
        recordContexts(model, items, true);

        double prevLoss = Double.MAX_VALUE;
        for (int i = 0; i < iterations; i++) {
            model.updateWithUsers(users);
            model.updateWithItems(items);
            Double loss = model.calculateLoss(users, items);
            Assert.assertNotNull(loss);
            Assert.assertTrue(loss < prevLoss);
            prevLoss = loss;
        }

        // assert that the user-item predictions after N iterations is identical to expected predictions
//        String expected = "makoto -> (toothpaste:3.000), (toothbrush:4.346), (shaver:2.530), \n" +
//                "takuya -> (toothpaste:4.998), (toothbrush:0.409), (shaver:-0.565), \n" +
//                "jackson -> (toothpaste:1.001), (toothbrush:2.556), (shaver:1.618), \n";
        String predictionString = generatePredictionString(model, users, items);
        System.out.println(predictionString);
//        Assert.assertEquals(predictionString, expected);
    }

    private static String generatePredictionString(CofactorModel model, List<CofactorizationUDTF.TrainingSample> users, List<CofactorizationUDTF.TrainingSample> items) {
        StringBuilder predicted = new StringBuilder();
        for (CofactorizationUDTF.TrainingSample user : users) {
            predicted.append(user.context).append(" -> ");
            for (CofactorizationUDTF.TrainingSample item : items) {
                Double score = model.predict(user.context, item.context);
                predicted.append("(")
                        .append(item.context)
                        .append(":")
                        .append(String.format("%.3f", score))
                        .append("), ");
            }
            predicted.append('\n');
        }
        return predicted.toString();
    }

    private static String mapToString(CofactorModel.Weights weights) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, double[]> entry : weights.entrySet()) {
            sb.append(entry.getKey() + ": " + arrayToString(entry.getValue(), 3) + ", ");
        }
        return sb.toString();
    }

    private static String arrayToString(double[] A, int decimals) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < A.length; i++) {
            sb.append(String.format("%." + decimals + "f", A[i]));
            if (i != A.length - 1) {
                sb.append(", ");
            }
        }
        sb.append(']');
        return sb.toString();
    }

    private static List<CofactorizationUDTF.TrainingSample> getItemSamples(boolean isExplicit) {
        List<CofactorizationUDTF.TrainingSample> samples = new ArrayList<>();
        if (isExplicit) {
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TOOTHBRUSH,
                            new Feature[]{new StringFeature(MAKOTO, 4.5d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHPASTE, 1.22d), new StringFeature(SHAVER, 1.22d)}));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TOOTHPASTE,
                            new Feature[]{new StringFeature(TAKUYA, 5.d), new StringFeature(MAKOTO, 3.d), new StringFeature(JACKSON, 1.d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 1.22d), new StringFeature(SHAVER, 1.35d)}));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            SHAVER,
                            new Feature[]{new StringFeature(JACKSON, 2.d), new StringFeature(MAKOTO, 2.3d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 1.22d), new StringFeature(TOOTHPASTE, 1.35d)}));
        } else {
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TOOTHBRUSH,
                            new Feature[]{new StringFeature(MAKOTO, 1.d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHPASTE, 1.22d), new StringFeature(SHAVER, 1.22d)}));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TOOTHPASTE,
                            new Feature[]{new StringFeature(TAKUYA, 1.d), new StringFeature(MAKOTO, 1.d), new StringFeature(JACKSON, 1.d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 1.22d), new StringFeature(SHAVER, 1.35d)}));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            SHAVER,
                            new Feature[]{new StringFeature(JACKSON, 1.d), new StringFeature(MAKOTO, 1.d)},
                            false,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 1.22d), new StringFeature(TOOTHPASTE, 1.35d)}));
        }
        return samples;
    }

    private static List<CofactorizationUDTF.TrainingSample> getUserSamples(boolean isExplicit) {
        List<CofactorizationUDTF.TrainingSample> samples = new ArrayList<>();
        if (isExplicit) {
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            MAKOTO,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 4.5d), new StringFeature(TOOTHPASTE, 3.d), new StringFeature(SHAVER, 2.3d)},
                            false,
                            null));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TAKUYA,
                            new Feature[]{new StringFeature(TOOTHPASTE, 5.d)},
                            false,
                            null));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            JACKSON,
                            new Feature[]{new StringFeature(TOOTHPASTE, 1.d), new StringFeature(SHAVER, 2.d)},
                            false,
                            null));
        } else {
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            MAKOTO,
                            new Feature[]{new StringFeature(TOOTHBRUSH, 1.d), new StringFeature(TOOTHPASTE, 1.d), new StringFeature(SHAVER, 1.d)},
                            false,
                            null));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            TAKUYA,
                            new Feature[]{new StringFeature(TOOTHPASTE, 1.d)},
                            false,
                            null));
            samples.add(
                    new CofactorizationUDTF.TrainingSample(
                            JACKSON,
                            new Feature[]{new StringFeature(TOOTHPASTE, 1.d), new StringFeature(SHAVER, 1.d)},
                            false,
                            null));
        }
        return samples;
    }

    private void recordContexts(CofactorModel model, List<CofactorizationUDTF.TrainingSample> samples, boolean isItem) throws HiveException {
        for (CofactorizationUDTF.TrainingSample sample : samples) {
            model.recordContext(sample.context, isItem);
        }
    }

    private static List<CofactorizationUDTF.TrainingSample> getSamples_userAsContext_allItemsInBeta() {
        List<CofactorizationUDTF.TrainingSample> samples = new ArrayList<>();
        samples.add(new CofactorizationUDTF.TrainingSample(
                TAKUYA,
                getSubset_itemFeatureVector_implicitFeedback(),
                false,
                null));
        return samples;
    }

    private static List<CofactorizationUDTF.TrainingSample> getSamples_itemAsContext_allUsersInTheta() {
        List<CofactorizationUDTF.TrainingSample> samples = new ArrayList<>();
        samples.add(new CofactorizationUDTF.TrainingSample(
                TOOTHBRUSH,
                getSubset_userFeatureVector_implicitFeedback(),
                false,
                getToothbrushSPPMIVector()));
        return samples;
    }

    private static List<CofactorizationUDTF.TrainingSample> getSamples_itemAsContext_oneUserNotInTheta() {
        List<CofactorizationUDTF.TrainingSample> samples = new ArrayList<>();
        samples.add(new CofactorizationUDTF.TrainingSample(
                TOOTHBRUSH,
                getSuperset_userFeatureVector_implicitFeedback(),
                false,
                null));
        return samples;
    }


    private static boolean matricesAreEqual(double[][] A, double[][] B) {
        if (A.length != B.length || A[0].length != B[0].length) {
            return false;
        }
        for (int r = 0; r < A.length; r++) {
            for (int c = 0; c < A[0].length; c++) {
                if (Math.abs(A[r][c] - B[r][c]) > EPSILON) {
                    return false;
                }
            }
        }
        return true;
    }

    private static CofactorModel.Weights getTestTheta() {
        CofactorModel.Weights weights = new CofactorModel.Weights();
        weights.put(MAKOTO, new double[]{0.8, -0.7});
        weights.put(TAKUYA, new double[]{-0.05, 1.7});
        weights.put(JACKSON, new double[]{1.8, -0.3});
        return weights;
    }

    private static CofactorModel.Weights getTestBeta() {
        CofactorModel.Weights weights = new CofactorModel.Weights();
        weights.put(TOOTHBRUSH, new double[]{0.5, 0.3});
        weights.put(TOOTHPASTE, new double[]{1.1, 0.9});
        weights.put(SHAVER, new double[]{-2.2, 1.6});
        return weights;
    }

    private static CofactorModel.Weights getTestGamma() {
        CofactorModel.Weights weights = new CofactorModel.Weights();
        weights.put(TOOTHBRUSH, new double[]{1.3, -0.2});
        weights.put(TOOTHPASTE, new double[]{1.6, 0.1});
        weights.put(SHAVER, new double[]{3.2, -0.4});
        return weights;
    }

    private static Object2DoubleMap<String> getTestBetaBias() {
        Object2DoubleMap<String> weights = new Object2DoubleArrayMap<>();
        weights.put(TOOTHBRUSH, 0.1);
        weights.put(TOOTHPASTE, -1.9);
        weights.put(SHAVER, 2.3);
        return weights;
    }

    private static Object2DoubleMap<String> getTestGammaBias() {
        Object2DoubleMap<String> weights = new Object2DoubleArrayMap<>();
        weights.put(TOOTHBRUSH, 3.4);
        weights.put(TOOTHPASTE, -0.5);
        weights.put(SHAVER, 1.1);
        return weights;
    }

    private static List<Feature> getSubset_itemFeatureList_explicitFeedback() {
        List<Feature> items = new ArrayList<>();
        items.add(new StringFeature(TOOTHBRUSH, 5.d));
        items.add(new StringFeature(SHAVER, 3.d));
        return items;
    }

    private static List<Feature> getToothbrushSPPMIList() {
        List<Feature> sppmi = new ArrayList<>();
        sppmi.add(new StringFeature(TOOTHPASTE, 1.22));
        sppmi.add(new StringFeature(SHAVER, 1.22));
        return sppmi;
    }

    private static Feature[] getToothbrushSPPMIVector() {
        Feature[] sppmi = new Feature[2];
        sppmi[0] = new StringFeature(TOOTHPASTE, 1.22);
        sppmi[1] = new StringFeature(SHAVER, 1.22);
        return sppmi;
    }

    private static Feature[] getSubset_itemFeatureVector_implicitFeedback() {
        Feature[] items = new Feature[2];
        items[0] = new StringFeature(TOOTHBRUSH, 1.d);
        items[1] = new StringFeature(SHAVER, 1.d);
        return items;
    }

    private static Feature[] getSubset_userFeatureVector_implicitFeedback() {
        // Makoto and Jackson both prefer a particular item
        Feature[] f = new Feature[2];
        f[0] = new StringFeature(MAKOTO, 1.d);
        f[1] = new StringFeature(JACKSON, 1.d);
        return f;
    }

    private static Feature[] getSuperset_userFeatureVector_implicitFeedback() {
        // Makoto, Jackson and Alien prefer a particular item
        Feature[] f = new Feature[3];
        f[0] = new StringFeature(MAKOTO, 1.d);
        f[1] = new StringFeature(JACKSON, 1.d);
        f[2] = new StringFeature(ALIEN, 1.d);
        assert !getTestGamma().containsKey(ALIEN);
        return f;
    }
}
