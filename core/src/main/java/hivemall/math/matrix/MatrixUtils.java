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

import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.matrix.ints.IntMatrix;
import hivemall.math.matrix.sparse.CSCMatrix;
import hivemall.math.matrix.sparse.CSRMatrix;
import hivemall.math.matrix.sparse.floats.CSCFloatMatrix;
import hivemall.math.matrix.sparse.floats.CSRFloatMatrix;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.mutable.MutableInt;

import java.util.Arrays;
import java.util.Comparator;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class MatrixUtils {

    private MatrixUtils() {}

    @Nonnull
    public static Matrix shuffle(@Nonnull final Matrix m, @Nonnull final int[] indices) {
        Preconditions.checkArgument(m.numRows() <= indices.length, "m.numRow() `" + m.numRows()
                + "` MUST be equals to or less than |swapIndicies| `" + indices.length + "`");

        final MatrixBuilder builder = m.builder();
        final VectorProcedure proc = new VectorProcedure() {
            public void apply(int col, double value) {
                builder.nextColumn(col, value);
            }
        };
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            m.eachNonNullInRow(idx, proc);
            builder.nextRow();
        }
        return builder.buildMatrix();
    }

    /**
     * Returns the index of maximum value of an array.
     * 
     * @return -1 if there are no columns
     */
    public static int whichMax(@Nonnull final IntMatrix matrix, @Nonnegative final int row) {
        final MutableInt m = new MutableInt(Integer.MIN_VALUE);
        final MutableInt which = new MutableInt(-1);
        matrix.eachInRow(row, new VectorProcedure() {
            @Override
            public void apply(int i, int value) {
                if (value > m.getValue()) {
                    m.setValue(value);
                    which.setValue(i);
                }
            }
        }, false);
        return which.getValue();
    }

    /**
     * @param data non-zero entries
     */
    @Nonnull
    public static CSRMatrix coo2csr(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final double[] data, @Nonnegative final int numRows,
            @Nonnegative final int numCols, final boolean sortColumns) {
        final int nnz = data.length;
        Preconditions.checkArgument(rows.length == nnz);
        Preconditions.checkArgument(cols.length == nnz);

        final int[] rowPointers = new int[numRows + 1];
        final int[] colIndicies = new int[nnz];
        final double[] values = new double[nnz];

        coo2csr(rows, cols, data, rowPointers, colIndicies, values, numRows, numCols, nnz);

        if (sortColumns) {
            sortIndicies(rowPointers, colIndicies, values);
        }
        return new CSRMatrix(rowPointers, colIndicies, values, numCols);
    }

    /**
     * @param data non-zero entries
     */
    @Nonnull
    public static CSRFloatMatrix coo2csr(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final float[] data, @Nonnegative final int numRows,
            @Nonnegative final int numCols, final boolean sortColumns) {
        final int nnz = data.length;
        Preconditions.checkArgument(rows.length == nnz);
        Preconditions.checkArgument(cols.length == nnz);

        final int[] rowPointers = new int[numRows + 1];
        final int[] colIndicies = new int[nnz];
        final float[] values = new float[nnz];

        coo2csr(rows, cols, data, rowPointers, colIndicies, values, numRows, numCols, nnz);

        if (sortColumns) {
            sortIndicies(rowPointers, colIndicies, values);
        }
        return new CSRFloatMatrix(rowPointers, colIndicies, values, numCols);
    }

    @Nonnull
    public static CSCMatrix coo2csc(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final double[] data, @Nonnegative final int numRows,
            @Nonnegative final int numCols, final boolean sortRows) {
        final int nnz = data.length;
        Preconditions.checkArgument(rows.length == nnz);
        Preconditions.checkArgument(cols.length == nnz);

        final int[] columnPointers = new int[numCols + 1];
        final int[] rowIndicies = new int[nnz];
        final double[] values = new double[nnz];

        coo2csr(cols, rows, data, columnPointers, rowIndicies, values, numCols, numRows, nnz);

        if (sortRows) {
            sortIndicies(columnPointers, rowIndicies, values);
        }

        return new CSCMatrix(columnPointers, rowIndicies, values, numRows, numCols);
    }

    @Nonnull
    public static CSCFloatMatrix coo2csc(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final float[] data, @Nonnegative final int numRows,
            @Nonnegative final int numCols, final boolean sortRows) {
        final int nnz = data.length;
        Preconditions.checkArgument(rows.length == nnz);
        Preconditions.checkArgument(cols.length == nnz);

        final int[] columnPointers = new int[numCols + 1];
        final int[] rowIndicies = new int[nnz];
        final float[] values = new float[nnz];

        coo2csr(cols, rows, data, columnPointers, rowIndicies, values, numCols, numRows, nnz);

        if (sortRows) {
            sortIndicies(columnPointers, rowIndicies, values);
        }

        return new CSCFloatMatrix(columnPointers, rowIndicies, values, numRows, numCols);
    }

    private static void coo2csr(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final double[] data, @Nonnull final int[] rowPointers,
            @Nonnull final int[] colIndicies, @Nonnull final double[] values,
            @Nonnegative final int numRows, @Nonnegative final int numCols, final int nnz) {
        // compute nnz per for each row to get rowPointers
        for (int n = 0; n < nnz; n++) {
            rowPointers[rows[n]]++;
        }
        for (int i = 0, sum = 0; i < numRows; i++) {
            int curr = rowPointers[i];
            rowPointers[i] = sum;
            sum += curr;
        }
        rowPointers[numRows] = nnz;

        // copy cols, data to colIndicies, csrValues
        for (int n = 0; n < nnz; n++) {
            int row = rows[n];
            int dst = rowPointers[row];

            colIndicies[dst] = cols[n];
            values[dst] = data[n];

            rowPointers[row]++;
        }

        for (int i = 0, last = 0; i <= numRows; i++) {
            int tmp = rowPointers[i];
            rowPointers[i] = last;
            last = tmp;
        }
    }

    private static void coo2csr(@Nonnull final int[] rows, @Nonnull final int[] cols,
            @Nonnull final float[] data, @Nonnull final int[] rowPointers,
            @Nonnull final int[] colIndicies, @Nonnull final float[] values,
            @Nonnegative final int numRows, @Nonnegative final int numCols, final int nnz) {
        // compute nnz per for each row to get rowPointers
        for (int n = 0; n < nnz; n++) {
            rowPointers[rows[n]]++;
        }
        for (int i = 0, sum = 0; i < numRows; i++) {
            int curr = rowPointers[i];
            rowPointers[i] = sum;
            sum += curr;
        }
        rowPointers[numRows] = nnz;

        // copy cols, data to colIndicies, csrValues
        for (int n = 0; n < nnz; n++) {
            int row = rows[n];
            int dst = rowPointers[row];

            colIndicies[dst] = cols[n];
            values[dst] = data[n];

            rowPointers[row]++;
        }

        for (int i = 0, last = 0; i <= numRows; i++) {
            int tmp = rowPointers[i];
            rowPointers[i] = last;
            last = tmp;
        }
    }

    private static void sortIndicies(@Nonnull final int[] rowPointers,
            @Nonnull final int[] colIndicies, @Nonnull final double[] values) {
        final int numRows = rowPointers.length - 1;
        if (numRows <= 1) {
            return;
        }

        for (int i = 0; i < numRows; i++) {
            final int rowStart = rowPointers[i];
            final int rowEnd = rowPointers[i + 1];

            final int numCols = rowEnd - rowStart;
            Preconditions.checkArgument(numCols > 0);

            final IntDoublePair[] pairs = new IntDoublePair[numCols];
            for (int jj = rowStart, n = 0; jj < rowEnd; jj++, n++) {
                pairs[n] = new IntDoublePair(colIndicies[jj], values[jj]);
            }

            Arrays.sort(pairs, new Comparator<IntDoublePair>() {
                @Override
                public int compare(IntDoublePair x, IntDoublePair y) {
                    return Integer.compare(x.key, y.key);
                }
            });

            for (int jj = rowStart, n = 0; jj < rowEnd; jj++, n++) {
                IntDoublePair tmp = pairs[n];
                colIndicies[jj] = tmp.key;
                values[jj] = tmp.value;
            }
        }
    }

    private static void sortIndicies(@Nonnull final int[] rowPointers,
            @Nonnull final int[] colIndicies, @Nonnull final float[] values) {
        final int numRows = rowPointers.length - 1;
        if (numRows <= 1) {
            return;
        }

        for (int i = 0; i < numRows; i++) {
            final int rowStart = rowPointers[i];
            final int rowEnd = rowPointers[i + 1];

            final int numCols = rowEnd - rowStart;
            Preconditions.checkArgument(numCols > 0);

            final IntFloatPair[] pairs = new IntFloatPair[numCols];
            for (int jj = rowStart, n = 0; jj < rowEnd; jj++, n++) {
                pairs[n] = new IntFloatPair(colIndicies[jj], values[jj]);
            }

            Arrays.sort(pairs, new Comparator<IntFloatPair>() {
                @Override
                public int compare(IntFloatPair x, IntFloatPair y) {
                    return Integer.compare(x.key, y.key);
                }
            });

            for (int jj = rowStart, n = 0; jj < rowEnd; jj++, n++) {
                IntFloatPair tmp = pairs[n];
                colIndicies[jj] = tmp.key;
                values[jj] = tmp.value;
            }
        }
    }

    private static final class IntDoublePair {

        final int key;
        final double value;

        IntDoublePair(int key, double value) {
            this.key = key;
            this.value = value;
        }
    }

    private static final class IntFloatPair {

        final int key;
        final float value;

        IntFloatPair(int key, float value) {
            this.key = key;
            this.value = value;
        }
    }

}
