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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class CofactorModelTest {
    private static final double EPSILON = 1e-8;

    @Test
    public void precomputeWTW() throws HiveException {
        Map<String, RealVector> weights = new HashMap<>();
        weights.put("toothbrush", new ArrayRealVector(new double[]{0.5, 0.3}));
        weights.put("toothpaste", new ArrayRealVector(new double[]{1.1, 0.9}));
        weights.put("shaver", new ArrayRealVector(new double[]{-2.2, 1.6}));

        RealMatrix expectedWTW = new Array2DRowRealMatrix(new double[][]{
                {0.63, -0.238},
                {-0.238, 0.346}
        });

        RealMatrix actualWTW = CofactorModel.computeWeightsTWeights(weights, 2, 0.1f);
        System.out.println(actualWTW.toString());
        Assert.assertTrue(matricesAreEqual(actualWTW, expectedWTW));
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

}
