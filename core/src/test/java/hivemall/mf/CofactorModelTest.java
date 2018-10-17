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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

public class CofactorModelTest {
    private static final double EPSILON = 1e-3;
    private static final int NUM_FACTORS = 2;
    private static final String TOOTHBRUSH = "toothbrush";
    private static final String TOOTHPASTE = "toothpaste";
    private static final String SHAVER = "shaver";
    private static final String MAKOTO = "makoto";
    private static final String TAKUYA = "takuya";
    private static final String JACKSON = "jackson";
    private static final double DUMMY_VALUE = 0.d;

    @Before

    @Test
    public void calculateWTW() throws HiveException {
        Map<String, RealVector> weights = getTestBeta();

        RealMatrix expectedWTW = new Array2DRowRealMatrix(new double[][]{
                {0.63, -0.238},
                {-0.238, 0.346}
        });

        RealMatrix actualWTW = CofactorModel.calculateWTW(weights, 2, 0.1f);
        Assert.assertTrue(matricesAreEqual(actualWTW, expectedWTW));
    }

    @Test
    public void calculateA() throws HiveException {
        Map<String, RealVector> weights = getTestBeta();
        List<Feature> items = getSubset_itemFeatureList_explicitFeedback();
        RealVector actual = CofactorModel.calculateA(items, weights, 0.5f);
        double[] expected = new double[]{-2.05, 3.15};
        Assert.assertArrayEquals(actual.toArray(), expected, EPSILON);
    }

    @Test
    public void calculateDelta() throws HiveException {
        Map<String, RealVector> weights = getTestBeta();
        List<Feature> items = getSubset_itemFeatureList_explicitFeedback();

        RealMatrix actual = CofactorModel.calculateDelta(items, weights, NUM_FACTORS, 0.9f);
        RealMatrix expected = new Array2DRowRealMatrix(new double[][]{
                {4.581, -3.033},
                {-3.033, 2.385}
        });

        Assert.assertTrue(matricesAreEqual(actual, expected));
    }

    @Test
    public void solve_updateUserWithImplicitFeedback() throws HiveException {
        final float c0 = 0.1f, c1 = 1.f, lambdaTheta = 1e-5f;
        Map<String, RealVector> weights = getTestBeta();
        RealMatrix identity = null;
        RealMatrix BTBpR = CofactorModel.calculateWTWpR(weights, NUM_FACTORS, c0, identity, lambdaTheta);

        List<Feature> items = getSubset_itemFeatureList_implicitFeedback();

        RealVector A = CofactorModel.calculateA(items, weights, c1);
        Assert.assertArrayEquals(A.toArray(), new double[]{-1.7, 1.9}, EPSILON);

        RealMatrix delta = CofactorModel.calculateDelta(items, weights, NUM_FACTORS, c1 - c0);
        RealMatrix B = BTBpR.add(delta);

        Assert.assertTrue(matricesAreEqual(B, new Array2DRowRealMatrix(new double[][]{
                {5.21101, -3.271},
                {-3.271, 2.73101}
        })));

        RealVector actual = CofactorModel.solve(B, A);
        RealVector expected = new ArrayRealVector(new double[]{0.44514062, 1.22886953});
        Assert.assertArrayEquals(actual.toArray(), expected.toArray(), EPSILON);
    }

    @Test
    public void calculateRSD() {

        Feature currentItem = new StringFeature(TOOTHBRUSH, DUMMY_VALUE);
        RealVector actual = CofactorModel.calculateRSD(
                currentItem,
                getToothbrushSPPMIList(),
                NUM_FACTORS,
                getTestBetaBias(),
                getTestGammaBias(),
                getTestGamma());
        double[] expected = new double[]{-1.12, 0.47};
        Assert.assertArrayEquals(actual.toArray(), expected, EPSILON);
    }

    @Test
    public void solve_updateOneItemWithImplicitFeedback() throws HiveException {
        final float c0 = 0.1f, c1 = 1.f, lambdaBeta = 1e-5f;
        RealMatrix identity = null;

        Map<String, Double> betaBias = getTestBetaBias();
        Map<String, Double> gammaBias = getTestGammaBias();
        Map<String, RealVector> gamma = getTestGamma();
        Map<String, RealVector> theta = getTestTheta();

        // solve for new weights for toothbrush
        Feature currentItem = new StringFeature(TOOTHBRUSH, DUMMY_VALUE);

        // get users who preferred / clicked / chose toothbrush (implicit rating)
        List<Feature> trainableUsers = getSubset_userFeatureList_implicitFeedback();

        RealMatrix TTTpR = CofactorModel.calculateWTWpR(theta, NUM_FACTORS, c0, identity, lambdaBeta);

        // get items that cooccur with toothbrush
        List<Feature> trainableCooccurringItems = getToothbrushSPPMIList();
        RealVector RSD = CofactorModel.calculateRSD(currentItem, trainableCooccurringItems, NUM_FACTORS, betaBias, gammaBias, gamma);
        RealVector ApRSD = CofactorModel.calculateA(trainableUsers, theta, c1).add(RSD);

        RealMatrix GTG = CofactorModel.calculateDelta(trainableCooccurringItems, gamma, NUM_FACTORS, 1.f);
        RealMatrix delta = CofactorModel.calculateDelta(trainableUsers, theta, NUM_FACTORS, c1 - c0);
        RealMatrix B = TTTpR.add(delta).add(GTG);

        // solve and update factors
        RealVector actual = CofactorModel.solve(B, ApRSD);

        RealVector expected = new ArrayRealVector(new double[]{0.02884247, -0.44823876});
        Assert.assertArrayEquals(actual.toArray(), expected.toArray(), EPSILON);
    }

    @Test
    public void calculateNewGammaVector() throws HiveException {
        final float lambdaGamma = 1e-5f;
        RealMatrix identity = null;

        Map<String, Double> betaBias = getTestBetaBias();
        Map<String, Double> gammaBias = getTestGammaBias();
        Map<String, RealVector> beta = getTestBeta();

        CofactorizationUDTF.TrainingSample currentItem = new CofactorizationUDTF.TrainingSample(
                new StringFeature(TOOTHBRUSH, DUMMY_VALUE), null, getToothbrushSPPMIVector());

        RealVector actual = CofactorModel.calculateNewGammaVector(currentItem, beta, gammaBias, betaBias, NUM_FACTORS, identity, lambdaGamma);
        RealVector expected = new ArrayRealVector(new double[]{0.95722067, -2.05881636});
        Assert.assertArrayEquals(actual.toArray(), expected.toArray(), EPSILON);
    }

    private static boolean matricesAreEqual(RealMatrix A, RealMatrix B) {
        double[][] dataA = A.getData(), dataB = B.getData();
        if (dataA.length != dataB.length || dataA[0].length != dataB[0].length) {
            return false;
        }
        for (int r = 0; r < dataA.length; r++) {
            for (int c = 0; c < dataA[0].length; c++) {
                if (Math.abs(dataA[r][c] - dataB[r][c]) > EPSILON) {
                    return false;
                }
            }
        }
        return true;
    }

    private static Map<String, RealVector> getTestTheta() {
        Map<String, RealVector> weights = new HashMap<>();
        weights.put(MAKOTO, new ArrayRealVector(new double[]{0.8, -0.7}));
        weights.put(TAKUYA, new ArrayRealVector(new double[]{-0.05, 1.7}));
        weights.put(JACKSON, new ArrayRealVector(new double[]{1.8, -0.3}));
        return weights;
    }

    private static Map<String, RealVector> getTestBeta() {
        Map<String, RealVector> weights = new HashMap<>();
        weights.put(TOOTHBRUSH, new ArrayRealVector(new double[]{0.5, 0.3}));
        weights.put(TOOTHPASTE, new ArrayRealVector(new double[]{1.1, 0.9}));
        weights.put(SHAVER, new ArrayRealVector(new double[]{-2.2, 1.6}));
        return weights;
    }

    private static Map<String, Double> getTestBetaBias() {
        Map<String, Double> weights = new HashMap<>();
        weights.put(TOOTHBRUSH, 0.1);
        weights.put(TOOTHPASTE, -1.9);
        weights.put(SHAVER, 2.3);
        return weights;
    }

    private static Map<String, Double> getTestGammaBias() {
        Map<String, Double> weights = new HashMap<>();
        weights.put(TOOTHBRUSH, 3.4);
        weights.put(TOOTHPASTE, -0.5);
        weights.put(SHAVER, 1.1);
        return weights;
    }

    private static Map<String, RealVector> getTestGamma() {
        Map<String, RealVector> weights = new HashMap<>();
        weights.put(TOOTHBRUSH, new ArrayRealVector(new double[]{1.3, -0.2}));
        weights.put(TOOTHPASTE, new ArrayRealVector(new double[]{1.6, 0.1}));
        weights.put(SHAVER, new ArrayRealVector(new double[]{3.2, -0.4}));
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
        sppmi.add(new StringFeature(TOOTHPASTE, 0.7));
        sppmi.add(new StringFeature(SHAVER, 0.3));
        return sppmi;
    }

    private static Feature[] getToothbrushSPPMIVector() {
        Feature[] sppmi = new Feature[2];
        sppmi[0] = new StringFeature(TOOTHPASTE, 0.7);
        sppmi[1] = new StringFeature(SHAVER, 0.3);
        return sppmi;
    }

    private static List<Feature> getSubset_itemFeatureList_implicitFeedback() {
        List<Feature> items = new ArrayList<>();
        items.add(new StringFeature(TOOTHBRUSH, 1.d));
        items.add(new StringFeature(SHAVER, 1.d));
        return items;
    }

    private static List<Feature> getSubset_userFeatureList_implicitFeedback() {
        // Makoto and Jackson both prefer a particular item
        List<Feature> users = new ArrayList<>();
        users.add(new StringFeature(MAKOTO, 1.d));
        users.add(new StringFeature(JACKSON, 1.d));
        return users;
    }

}
