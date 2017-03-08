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
package hivemall.matrix;

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

        matrix.setDefaultValue(Double.NaN);
        Assert.assertEquals(Double.NaN, matrix.get(5, 4), 0.d);
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

        matrix.setDefaultValue(Double.NaN);
        Assert.assertEquals(Double.NaN, matrix.get(5, 4), 0.d);
    }

    @Test
    public void testReadOnlyCSRMatrixNoRow() {
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024, true);
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
    public void testReadOnlyDenseMatrix2d() {
        Matrix matrix = denseMatrix();
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
        Matrix matrix = new DenseMatrixBuilder(1024, true).buildMatrix();
        Assert.assertEquals(0, matrix.numRows());
        Assert.assertEquals(0, matrix.numColumns());
    }

    @Test(expected = UnsupportedOperationException.class)
    public void testReadOnlyDenseMatrix2dFailToChangeDefaultValue() {
        Matrix matrix = denseMatrix();
        matrix.setDefaultValue(Double.NaN);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyDenseMatrix2dFailOutOfBound1() {
        Matrix matrix = denseMatrix();
        matrix.get(7, 5);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testReadOnlyDenseMatrix2dFailOutOfBound2() {
        Matrix matrix = denseMatrix();
        matrix.get(6, 7);
    }

    @Test
    public void testCSRMatrixNullRow() {
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024, true);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextRow();
        builder.nextColumn(3, 66).nextRow();
        Matrix matrix = builder.buildMatrix();
        Assert.assertEquals(4, matrix.numRows());
    }

    private static Matrix csrMatrix() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024, true);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextColumn(2, 33).nextColumn(3, 34).nextColumn(4, 35).nextColumn(5, 36).nextRow();
        builder.nextColumn(3, 44).nextColumn(4, 45).nextRow();
        builder.nextColumn(5, 56).nextRow();
        builder.nextColumn(5, 66).nextRow();
        return builder.buildMatrix();
    }

    private static Matrix csrMatrixFromLibSVM() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        CSRMatrixBuilder builder = new CSRMatrixBuilder(1024, true);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }

    private static Matrix denseMatrix() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        DenseMatrixBuilder builder = new DenseMatrixBuilder(1024, true);
        builder.nextRow(new double[] {11, 12, 13, 14});
        builder.nextRow(new double[] {0, 22, 23});
        builder.nextRow(new double[] {0, 0, 33, 34, 35, 36});
        builder.nextRow(new double[] {0, 0, 0, 44, 45});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 56});
        builder.nextRow(new double[] {0, 0, 0, 0, 0, 66});
        return builder.buildMatrix();
    }

    private static Matrix denseMatrixSparseInput() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        DenseMatrixBuilder builder = new DenseMatrixBuilder(1024, true);
        builder.nextColumn(0, 11).nextColumn(1, 12).nextColumn(2, 13).nextColumn(3, 14).nextRow();
        builder.nextColumn(1, 22).nextColumn(2, 23).nextRow();
        builder.nextColumn(2, 33).nextColumn(3, 34).nextColumn(4, 35).nextColumn(5, 36).nextRow();
        builder.nextColumn(3, 44).nextColumn(4, 45).nextRow();
        builder.nextColumn(5, 56).nextRow();
        builder.nextColumn(5, 66).nextRow();
        return builder.buildMatrix();
    }

    private static Matrix denseMatrixFromLibSVM() {
        DenseMatrixBuilder builder = new DenseMatrixBuilder(1024, true);
        builder.nextRow(new String[] {"0:11", "1:12", "2:13", "3:14"});
        builder.nextRow(new String[] {"1:22", "2:23"});
        builder.nextRow(new String[] {"2:33", "3:34", "4:35", "5:36"});
        builder.nextRow(new String[] {"3:44", "4:45"});
        builder.nextRow(new String[] {"5:56"});
        builder.nextRow(new String[] {"5:66"});
        return builder.buildMatrix();
    }

}
