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

import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.RowMajorMatrix;
import hivemall.math.matrix.builders.CSCMatrixBuilder;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.ColumnMajorDenseMatrixBuilder;
import hivemall.math.matrix.builders.DoKMatrixBuilder;
import hivemall.math.matrix.builders.RowMajorDenseMatrixBuilder;
import hivemall.math.matrix.dense.ColumnMajorDenseMatrix2d;
import hivemall.math.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.math.matrix.sparse.CSCMatrix;
import hivemall.math.matrix.sparse.CSRMatrix;
import hivemall.math.matrix.sparse.DoKMatrix;

import org.junit.Assert;
import org.junit.Test;

public class MatrixBuilderTest {

    @Test
    public void testReadOnlyCSRMatrix() {
        Matrix matrix = csrMatrix();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(2, matrix.numColumns(1));
        Assert.assertEquals(4, matrix.numColumns(2));
        Assert.assertEquals(2, matrix.numColumns(3));
        Assert.assertEquals(1, matrix.numColumns(4));
        Assert.assertEquals(1, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, matrix.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, matrix.get(5, 4, Double.NaN), 0.d);
    }

    @Test
    public void testReadOnlyCSRMatrixFromLibSVM() {
        Matrix matrix = csrMatrixFromLibSVM();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(2, matrix.numColumns(1));
        Assert.assertEquals(4, matrix.numColumns(2));
        Assert.assertEquals(2, matrix.numColumns(3));
        Assert.assertEquals(1, matrix.numColumns(4));
        Assert.assertEquals(1, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, matrix.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, matrix.get(5, 4, Double.NaN), 0.d);
    }

    @Test
    public void testReadOnlyCSRMatrixNoRow() {
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024);
        Matrix matrix = builder.buildMatrix();
        Assert.assertEquals(0, matrix.numRows());
        Assert.assertEquals(0, matrix.numColumns());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyCSRMatrixGetFail1() {
        Matrix matrix = csrMatrix();
        matrix.get(7, 5);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyCSRMatrixGetFail2() {
        Matrix matrix = csrMatrix();
        matrix.get(6, 7);
    }

    @Test
    public void testCSCMatrixFromLibSVM() {
        CSCMatrix matrix = cscMatrixFromLibSVM();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(2, matrix.numColumns(1));
        Assert.assertEquals(4, matrix.numColumns(2));
        Assert.assertEquals(2, matrix.numColumns(3));
        Assert.assertEquals(1, matrix.numColumns(4));
        Assert.assertEquals(1, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, matrix.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, matrix.get(5, 4, Double.NaN), 0.d);
    }

    @Test
    public void testCSC2CSR() {
        CSCMatrix csc = cscMatrixFromLibSVM();
        RowMajorMatrix csr = csc.toRowMajorMatrix();
        Assert.assertTrue(csr instanceof CSRMatrix);
        Assert.assertEquals(6, csr.numRows());
        Assert.assertEquals(6, csr.numColumns());
        Assert.assertEquals(4, csr.numColumns(0));
        Assert.assertEquals(2, csr.numColumns(1));
        Assert.assertEquals(4, csr.numColumns(2));
        Assert.assertEquals(2, csr.numColumns(3));
        Assert.assertEquals(1, csr.numColumns(4));
        Assert.assertEquals(1, csr.numColumns(5));

        Assert.assertEquals(11d, csr.get(0, 0), 0.d);
        Assert.assertEquals(12d, csr.get(0, 1), 0.d);
        Assert.assertEquals(13d, csr.get(0, 2), 0.d);
        Assert.assertEquals(14d, csr.get(0, 3), 0.d);
        Assert.assertEquals(22d, csr.get(1, 1), 0.d);
        Assert.assertEquals(23d, csr.get(1, 2), 0.d);
        Assert.assertEquals(33d, csr.get(2, 2), 0.d);
        Assert.assertEquals(34d, csr.get(2, 3), 0.d);
        Assert.assertEquals(35d, csr.get(2, 4), 0.d);
        Assert.assertEquals(36d, csr.get(2, 5), 0.d);
        Assert.assertEquals(44d, csr.get(3, 3), 0.d);
        Assert.assertEquals(45d, csr.get(3, 4), 0.d);
        Assert.assertEquals(56d, csr.get(4, 5), 0.d);
        Assert.assertEquals(66d, csr.get(5, 5), 0.d);

        Assert.assertEquals(0.d, csr.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, csr.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, csr.get(5, 4, Double.NaN), 0.d);
    }

    @Test
    public void testCSC2CSR2CSR() {
        CSCMatrix csc = cscMatrixFromLibSVM();
        CSCMatrix csc2 = csc.toRowMajorMatrix().toColumnMajorMatrix();
        Assert.assertEquals(csc.nnz(), csc2.nnz());
        Assert.assertEquals(6, csc2.numRows());
        Assert.assertEquals(6, csc2.numColumns());
        Assert.assertEquals(4, csc2.numColumns(0));
        Assert.assertEquals(2, csc2.numColumns(1));
        Assert.assertEquals(4, csc2.numColumns(2));
        Assert.assertEquals(2, csc2.numColumns(3));
        Assert.assertEquals(1, csc2.numColumns(4));
        Assert.assertEquals(1, csc2.numColumns(5));

        Assert.assertEquals(11d, csc2.get(0, 0), 0.d);
        Assert.assertEquals(12d, csc2.get(0, 1), 0.d);
        Assert.assertEquals(13d, csc2.get(0, 2), 0.d);
        Assert.assertEquals(14d, csc2.get(0, 3), 0.d);
        Assert.assertEquals(22d, csc2.get(1, 1), 0.d);
        Assert.assertEquals(23d, csc2.get(1, 2), 0.d);
        Assert.assertEquals(33d, csc2.get(2, 2), 0.d);
        Assert.assertEquals(34d, csc2.get(2, 3), 0.d);
        Assert.assertEquals(35d, csc2.get(2, 4), 0.d);
        Assert.assertEquals(36d, csc2.get(2, 5), 0.d);
        Assert.assertEquals(44d, csc2.get(3, 3), 0.d);
        Assert.assertEquals(45d, csc2.get(3, 4), 0.d);
        Assert.assertEquals(56d, csc2.get(4, 5), 0.d);
        Assert.assertEquals(66d, csc2.get(5, 5), 0.d);

        Assert.assertEquals(0.d, csc2.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, csc2.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, csc2.get(5, 4, Double.NaN), 0.d);
    }


    @Test
    public void testDoKMatrixFromLibSVM() {
        Matrix matrix = dokMatrixFromLibSVM();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(2, matrix.numColumns(1));
        Assert.assertEquals(4, matrix.numColumns(2));
        Assert.assertEquals(2, matrix.numColumns(3));
        Assert.assertEquals(1, matrix.numColumns(4));
        Assert.assertEquals(1, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);
        Assert.assertEquals(-1.d, matrix.get(5, 4, -1.d), 0.d);

        Assert.assertEquals(Double.NaN, matrix.get(5, 4, Double.NaN), 0.d);
    }

    @Test
    public void testReadOnlyDenseMatrix2d() {
        Matrix matrix = rowMajorDenseMatrix();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(3, matrix.numColumns(1));
        Assert.assertEquals(6, matrix.numColumns(2));
        Assert.assertEquals(5, matrix.numColumns(3));
        Assert.assertEquals(6, matrix.numColumns(4));
        Assert.assertEquals(6, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
    }

    @Test
    public void testReadOnlyDenseMatrix2dSparseInput() {
        Matrix matrix = denseMatrixSparseInput();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(3, matrix.numColumns(1));
        Assert.assertEquals(6, matrix.numColumns(2));
        Assert.assertEquals(5, matrix.numColumns(3));
        Assert.assertEquals(6, matrix.numColumns(4));
        Assert.assertEquals(6, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
    }

    @Test
    public void testReadOnlyDenseMatrix2dFromLibSVM() {
        Matrix matrix = denseMatrixFromLibSVM();
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());
        Assert.assertEquals(4, matrix.numColumns(0));
        Assert.assertEquals(3, matrix.numColumns(1));
        Assert.assertEquals(6, matrix.numColumns(2));
        Assert.assertEquals(5, matrix.numColumns(3));
        Assert.assertEquals(6, matrix.numColumns(4));
        Assert.assertEquals(6, matrix.numColumns(5));

        Assert.assertEquals(11d, matrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, matrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, matrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, matrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, matrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, matrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, matrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, matrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, matrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, matrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, matrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, matrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, matrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, matrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, matrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, matrix.get(1, 0), 0.d);
    }

    @Test
    public void testReadOnlyDenseMatrix2dNoRow() {
        Matrix matrix = new RowMajorDenseMatrixBuilder(1024).buildMatrix();
        Assert.assertEquals(0, matrix.numRows());
        Assert.assertEquals(0, matrix.numColumns());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyDenseMatrix2dFailOutOfBound1() {
        Matrix matrix = rowMajorDenseMatrix();
        matrix.get(7, 5);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyDenseMatrix2dFailOutOfBound2() {
        Matrix matrix = rowMajorDenseMatrix();
        matrix.get(6, 7);
    }

    @Test
    public void testColumnMajorDenseMatrix2d() {
        ColumnMajorDenseMatrix2d colMatrix = columnMajorDenseMatrix();

        Assert.assertEquals(6, colMatrix.numRows());
        Assert.assertEquals(6, colMatrix.numColumns());
        Assert.assertEquals(4, colMatrix.numColumns(0));
        Assert.assertEquals(2, colMatrix.numColumns(1));
        Assert.assertEquals(4, colMatrix.numColumns(2));
        Assert.assertEquals(2, colMatrix.numColumns(3));
        Assert.assertEquals(1, colMatrix.numColumns(4));
        Assert.assertEquals(1, colMatrix.numColumns(5));

        Assert.assertEquals(11d, colMatrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, colMatrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, colMatrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, colMatrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, colMatrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, colMatrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, colMatrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, colMatrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, colMatrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, colMatrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, colMatrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, colMatrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, colMatrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, colMatrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, colMatrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, colMatrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, colMatrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, colMatrix.get(1, 0), 0.d);
    }

    @Test
    public void testDenseMatrixColumnMajor2RowMajor() {
        ColumnMajorDenseMatrix2d colMatrix = columnMajorDenseMatrix();
        RowMajorDenseMatrix2d rowMatrix = colMatrix.toRowMajorMatrix();

        Assert.assertEquals(6, rowMatrix.numRows());
        Assert.assertEquals(6, rowMatrix.numColumns());
        Assert.assertEquals(4, rowMatrix.numColumns(0));
        Assert.assertEquals(3, rowMatrix.numColumns(1));
        Assert.assertEquals(6, rowMatrix.numColumns(2));
        Assert.assertEquals(5, rowMatrix.numColumns(3));
        Assert.assertEquals(6, rowMatrix.numColumns(4));
        Assert.assertEquals(6, rowMatrix.numColumns(5));

        Assert.assertEquals(11d, rowMatrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, rowMatrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, rowMatrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, rowMatrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, rowMatrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, rowMatrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, rowMatrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, rowMatrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, rowMatrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, rowMatrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, rowMatrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, rowMatrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, rowMatrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, rowMatrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, rowMatrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, rowMatrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, rowMatrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, rowMatrix.get(1, 0), 0.d);

        // convert back to column major matrix

        colMatrix = rowMatrix.toColumnMajorMatrix();

        Assert.assertEquals(6, colMatrix.numRows());
        Assert.assertEquals(6, colMatrix.numColumns());
        Assert.assertEquals(4, colMatrix.numColumns(0));
        Assert.assertEquals(2, colMatrix.numColumns(1));
        Assert.assertEquals(4, colMatrix.numColumns(2));
        Assert.assertEquals(2, colMatrix.numColumns(3));
        Assert.assertEquals(1, colMatrix.numColumns(4));
        Assert.assertEquals(1, colMatrix.numColumns(5));

        Assert.assertEquals(11d, colMatrix.get(0, 0), 0.d);
        Assert.assertEquals(12d, colMatrix.get(0, 1), 0.d);
        Assert.assertEquals(13d, colMatrix.get(0, 2), 0.d);
        Assert.assertEquals(14d, colMatrix.get(0, 3), 0.d);
        Assert.assertEquals(22d, colMatrix.get(1, 1), 0.d);
        Assert.assertEquals(23d, colMatrix.get(1, 2), 0.d);
        Assert.assertEquals(33d, colMatrix.get(2, 2), 0.d);
        Assert.assertEquals(34d, colMatrix.get(2, 3), 0.d);
        Assert.assertEquals(35d, colMatrix.get(2, 4), 0.d);
        Assert.assertEquals(36d, colMatrix.get(2, 5), 0.d);
        Assert.assertEquals(44d, colMatrix.get(3, 3), 0.d);
        Assert.assertEquals(45d, colMatrix.get(3, 4), 0.d);
        Assert.assertEquals(56d, colMatrix.get(4, 5), 0.d);
        Assert.assertEquals(66d, colMatrix.get(5, 5), 0.d);

        Assert.assertEquals(0.d, colMatrix.get(5, 4), 0.d);

        Assert.assertEquals(0.d, colMatrix.get(1, 0), 0.d);
        Assert.assertEquals(0.d, colMatrix.get(1, 3), 0.d);
        Assert.assertEquals(0.d, colMatrix.get(1, 0), 0.d);
    }

    @Test
    public void testCSRMatrixNullRow() {
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextRow();
        builder.nextColumn(3, 66).nextRow();
        Matrix matrix = builder.buildMatrix();
        Assert.assertEquals(4, matrix.numRows());
    }

    private static CSRMatrix csrMatrix() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextColumn(2, 33).nextColumn(3, 34).nextColumn(4, 35).nextColumn(5, 36).nextRow();
        builder.nextColumn(3, 44).nextColumn(4, 45).nextRow();
        builder.nextColumn(5, 56).nextRow();
        builder.nextColumn(5, 66).nextRow();
        return builder.buildMatrix();
    }

    private static CSRMatrix csrMatrixFromLibSVM() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }

    private static CSCMatrix cscMatrixFromLibSVM() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        CSCMatrixBuilder builder = new CSCMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }


    private static DoKMatrix dokMatrixFromLibSVM() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        DoKMatrixBuilder builder = new DoKMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }

    private static RowMajorDenseMatrix2d rowMajorDenseMatrix() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        RowMajorDenseMatrixBuilder builder = new RowMajorDenseMatrixBuilder(1024);
        builder.nextRow(new double[] {11, 12, 13, 14});
        builder.nextRow(new double[] {0, 22, 23});
        builder.nextRow(new double[] {0, 0, 33, 34, 35, 36});
        builder.nextRow(new double[] {0, 0, 0, 44, 45});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 56});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 66});
        return builder.buildMatrix();
    }

    private static ColumnMajorDenseMatrix2d columnMajorDenseMatrix() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        ColumnMajorDenseMatrixBuilder builder = new ColumnMajorDenseMatrixBuilder(1024);
        builder.nextRow(new double[] {11, 12, 13, 14});
        builder.nextRow(new double[] {0, 22, 23});
        builder.nextRow(new double[] {0, 0, 33, 34, 35, 36});
        builder.nextRow(new double[] {0, 0, 0, 44, 45});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 56});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 66});
        return builder.buildMatrix();
    }

    private static RowMajorDenseMatrix2d denseMatrixSparseInput() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        RowMajorDenseMatrixBuilder builder = new RowMajorDenseMatrixBuilder(1024);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextColumn(2, 33).nextColumn(3, 34).nextColumn(4, 35).nextColumn(5, 36).nextRow();
        builder.nextColumn(3, 44).nextColumn(4, 45).nextRow();
        builder.nextColumn(5, 56).nextRow();
        builder.nextColumn(5, 66).nextRow();
        return builder.buildMatrix();
    }

    private static RowMajorDenseMatrix2d denseMatrixFromLibSVM() {
        RowMajorDenseMatrixBuilder builder = new RowMajorDenseMatrixBuilder(1024);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }

}
