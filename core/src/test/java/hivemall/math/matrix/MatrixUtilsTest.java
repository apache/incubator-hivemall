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
package hivemall.math.matrix;

import hivemall.math.matrix.sparse.CSCMatrix;
import hivemall.math.matrix.sparse.CSRMatrix;
import hivemall.utils.lang.ArrayUtils;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class MatrixUtilsTest {

    @Test
    public void testCoo2csr() {
        // 10  0  0  0 -2  0
        // 3  9  0  0  0  3
        // 0  7  8  7  0  0
        // 3  0  8  7  5  0
        // 0  8  0  9  9 13
        // 0  4  0  0  2 -1 
        double[] row1 = new double[] {10, 0, 0, 0, -2, 0};
        double[] row2 = new double[] {3, 9, 0, 0, 0, 3};
        double[] row3 = new double[] {0, 7, 8, 7, 0, 0};
        double[] row4 = new double[] {3, 0, 8, 7, 5, 0};
        double[] row5 = new double[] {0, 8, 0, 9, 9, 13};
        double[] row6 = new double[] {0, 4, 0, 0, 2, -1};
        double[][] matrix = new double[][] {row1, row2, row3, row4, row5, row6};

        int[] rows = new int[] {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
        int[] cols = new int[] {0, 4, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5};
        double[] data = new double[] {10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1};

        CSRMatrix matrix1 = MatrixUtils.coo2csr(rows, cols, data, 6, 6, false);
        Assert.assertEquals(data.length, matrix1.nnz());

        final Random rnd = new Random(43L);
        for (int i = data.length; i > 1; i--) {
            int to = rnd.nextInt(i);
            int from = i - 1;
            ArrayUtils.swap(rows, from, to);
            ArrayUtils.swap(cols, from, to);
            ArrayUtils.swap(data, from, to);
        }

        CSRMatrix matrix2 = MatrixUtils.coo2csr(rows, cols, data, 6, 6, true);
        Assert.assertEquals(data.length, matrix2.nnz());

        double[] dst1 = matrix1.row();
        Assert.assertEquals(6, matrix1.numRows());
        for (int row = 0; row < matrix1.numRows(); row++) {
            matrix1.getRow(row, dst1);
            Assert.assertArrayEquals(matrix[row], dst1, 1E-7d);
        }

        double[] dst2 = matrix2.row();
        Assert.assertEquals(6, matrix2.numRows());
        for (int row = 0; row < matrix2.numRows(); row++) {
            matrix2.getRow(row, dst2);
            Assert.assertArrayEquals(matrix[row], dst2, 1E-7d);
        }
    }

    @Test
    public void testCoo2csc() {
        // 10  0  0  0 -2  0
        // 3  9  0  0  0  3
        // 0  7  8  7  0  0
        // 3  0  8  7  5  0
        // 0  8  0  9  9 13
        // 0  4  0  0  2 -1 
        double[] row1 = new double[] {10, 0, 0, 0, -2, 0};
        double[] row2 = new double[] {3, 9, 0, 0, 0, 3};
        double[] row3 = new double[] {0, 7, 8, 7, 0, 0};
        double[] row4 = new double[] {3, 0, 8, 7, 5, 0};
        double[] row5 = new double[] {0, 8, 0, 9, 9, 13};
        double[] row6 = new double[] {0, 4, 0, 0, 2, -1};
        double[][] matrix = new double[][] {row1, row2, row3, row4, row5, row6};

        int[] rows = new int[] {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
        int[] cols = new int[] {0, 4, 0, 1, 5, 1, 2, 3, 0, 2, 3, 4, 1, 3, 4, 5, 1, 4, 5};
        double[] data = new double[] {10, -2, 3, 9, 3, 7, 8, 7, 3, 8, 7, 5, 8, 9, 9, 13, 4, 2, -1};

        CSCMatrix matrix1 = MatrixUtils.coo2csc(rows, cols, data, 6, 6, false);
        Assert.assertEquals(data.length, matrix1.nnz());

        final Random rnd = new Random(43L);
        for (int i = data.length; i > 1; i--) {
            int to = rnd.nextInt(i);
            int from = i - 1;
            ArrayUtils.swap(rows, from, to);
            ArrayUtils.swap(cols, from, to);
            ArrayUtils.swap(data, from, to);
        }

        CSCMatrix matrix2 = MatrixUtils.coo2csc(rows, cols, data, 6, 6, true);
        Assert.assertEquals(data.length, matrix2.nnz());

        double[] dst1 = matrix1.row();
        Assert.assertEquals(6, matrix1.numRows());
        for (int row = 0; row < matrix1.numRows(); row++) {
            matrix1.getRow(row, dst1);
            Assert.assertArrayEquals(matrix[row], dst1, 1E-7d);
        }

        double[] dst2 = matrix2.row();
        Assert.assertEquals(6, matrix2.numRows());
        for (int row = 0; row < matrix2.numRows(); row++) {
            matrix2.getRow(row, dst2);
            Assert.assertArrayEquals(matrix[row], dst2, 1E-7d);
        }
    }

}
