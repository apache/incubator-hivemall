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

    @Before

    @Test
    public void precomputeWTW() throws HiveException {
        Map<String, RealVector> weights = getTestWeights();

        RealMatrix expectedWTW = new Array2DRowRealMatrix(new double[][]{
                {0.63, -0.238},
                {-0.238, 0.346}
        });

        RealMatrix actualWTW = CofactorModel.calculateWTW(weights, 2, 0.1f);
        Assert.assertTrue(matricesAreEqual(actualWTW, expectedWTW));
    }

    @Test
    public void calculateA() throws HiveException {
        Map<String, RealVector> weights = getTestWeights();
        List<Feature> items = getSubsetFeatureList_explicitFeedback();
        RealVector actual = CofactorModel.calculateA(items, weights, 0.5f);
        double[] expected = new double[]{-2.05, 3.15};
        Assert.assertArrayEquals(actual.toArray(), expected, EPSILON);
    }

    @Test
    public void calculateDelta() throws HiveException {
        Map<String, RealVector> weights = getTestWeights();
        List<Feature> items = getSubsetFeatureList_explicitFeedback();

        RealMatrix actual = CofactorModel.calculateDelta(items, weights, NUM_FACTORS, 0.9f);
        RealMatrix expected = new Array2DRowRealMatrix(new double[][]{
                { 4.581, -3.033},
                { -3.033, 2.385}
        });

        Assert.assertTrue(matricesAreEqual(actual, expected));
    }

    @Test
    public void solve_implicitFeedback() throws HiveException {
        final float c0 = 0.1f, c1 = 1.f, lambdaTheta = 1e-5f;
        Map<String, RealVector> weights = getTestWeights();
        RealMatrix identity = null;
        RealMatrix BTBpR = CofactorModel.calculateWTWpR(weights, NUM_FACTORS, c0, identity, lambdaTheta);

        List<Feature> items = getSubsetFeatureList_implicitFeedback();

        RealVector A = CofactorModel.calculateA(items, weights, c1);
        Assert.assertArrayEquals(A.toArray(), new double[]{-1.7,  1.9}, EPSILON);

        RealMatrix delta = CofactorModel.calculateDelta(items, weights, NUM_FACTORS, c1 - c0);
        RealMatrix B = BTBpR.add(delta);

        Assert.assertTrue(matricesAreEqual(B, new Array2DRowRealMatrix(new double[][]{
                { 5.21101, -3.271  },
                {-3.271  ,  2.73101}
        })));

        RealVector actual = CofactorModel.solve(B, A);
        RealVector expected = new ArrayRealVector(new double[]{0.44514062, 1.22886953});
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

    private static Map<String, RealVector> getTestWeights() {
        Map<String, RealVector> weights = new HashMap<>();
        weights.put(TOOTHBRUSH, new ArrayRealVector(new double[]{0.5, 0.3}));
        weights.put(TOOTHPASTE, new ArrayRealVector(new double[]{1.1, 0.9}));
        weights.put(SHAVER, new ArrayRealVector(new double[]{-2.2, 1.6}));
        return weights;
    }

    private static List<Feature> getSubsetFeatureList_explicitFeedback() {
        List<Feature> items = new ArrayList<>();
        items.add(new StringFeature(TOOTHBRUSH, 5.d));
        items.add(new StringFeature(SHAVER, 3.d));
        return items;
    }

    private static List<Feature> getSubsetFeatureList_implicitFeedback() {
        List<Feature> items = new ArrayList<>();
        items.add(new StringFeature(TOOTHBRUSH, 1.d));
        items.add(new StringFeature(SHAVER, 1.d));
        return items;
    }

}
