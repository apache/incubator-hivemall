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
package hivemall.xgboost.utils;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import org.junit.Assert;
import org.junit.Test;

public class DMatrixBuilderTest {

    @Test
    public void testDenseMatrix() throws XGBoostError {
        DMatrix matrix = createDenseDMatrix();
        Assert.assertEquals(6, matrix.rowNum());
        matrix.dispose();
    }

    @Test
    public void testSparseMatrix() throws XGBoostError {
        DMatrix matrix = createSparseDMatrix();
        Assert.assertEquals(6, matrix.rowNum());
        matrix.dispose();
    }

    @Test
    public void testCreateFromCSREx() throws XGBoostError {
        // sparse matrix
        // 1 0 2 3 0
        // 4 0 2 3 5
        // 3 1 2 5 0
        DenseDMatrixBuilder builder = new DenseDMatrixBuilder(1024);
        builder.nextRow(new float[] {1, 0, 2, 3, 0});
        builder.nextRow(new float[] {4, 0, 2, 3, 5});
        builder.nextRow(new float[] {3, 1, 2, 5, 0});
        float[] label1 = new float[] {1, 0, 1};
        DMatrix dmat1 = builder.buildMatrix(label1);

        Assert.assertEquals(3, dmat1.rowNum());
        float[] label2 = dmat1.getLabel();
        Assert.assertArrayEquals(label1, label2, 0.f);
    }

    private static DMatrix createDenseDMatrix() throws XGBoostError {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        DenseDMatrixBuilder builder = new DenseDMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});

        float[] labels = new float[6];
        return builder.buildMatrix(labels);
    }

    private static DMatrix createSparseDMatrix() throws XGBoostError {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        SparseDMatrixBuilder builder = new SparseDMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});

        float[] labels = new float[6];
        return builder.buildMatrix(labels);
    }

}
