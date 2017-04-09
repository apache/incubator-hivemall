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
package hivemall.math.matrix.sparse;

import hivemall.math.matrix.RowMajorMatrix;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.lang.Preconditions;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Read-only CSR double Matrix.
 * 
 * @link http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 * @link http://www.cs.colostate.edu/~mcrob/toolbox/c++/sparseMatrix/sparse_matrix_compression.html
 */
public final class CSRMatrix extends RowMajorMatrix {

    @Nonnull
    private final int[] rowPointers;
    @Nonnull
    private final int[] columnIndices;
    @Nonnull
    private final double[] values;

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;
    @Nonnegative
    private final int nnz;

    public CSRMatrix(@Nonnull int[] rowPointers, @Nonnull int[] columnIndices,
            @Nonnull double[] values, @Nonnegative int numColumns) {
        super();
        Preconditions.checkArgument(rowPointers.length >= 1,
            "rowPointers must be greather than 0: " + rowPointers.length);
        Preconditions.checkArgument(columnIndices.length == values.length, "#columnIndices ("
                + columnIndices.length + ") must be equals to #values (" + values.length + ")");
        this.rowPointers = rowPointers;
        this.columnIndices = columnIndices;
        this.values = values;
        this.numRows = rowPointers.length - 1;
        this.numColumns = numColumns;
        this.nnz = values.length;
    }

    @Override
    public boolean isSparse() {
        return true;
    }

    @Override
    public boolean readOnly() {
        return true;
    }

    @Override
    public boolean swappable() {
        return false;
    }

    @Override
    public int nnz() {
        return nnz;
    }

    @Override
    public int numRows() {
        return numRows;
    }

    @Override
    public int numColumns() {
        return numColumns;
    }

    @Override
    public int numColumns(@Nonnegative final int row) {
        checkRowIndex(row, numRows);

        int columns = rowPointers[row + 1] - rowPointers[row];
        return columns;
    }

    @Override
    public double[] getRow(@Nonnegative final int index) {
        final double[] row = new double[numColumns];
        eachNonZeroInRow(index, new VectorProcedure() {
            public void apply(int col, double value) {
                row[col] = value;
            }
        });
        return row;
    }

    @Override
    public double[] getRow(@Nonnegative final int index, @Nonnull final double[] dst) {
        Arrays.fill(dst, 0.d);
        eachNonZeroInRow(index, new VectorProcedure() {
            public void apply(int col, double value) {
                checkColIndex(col, numColumns);
                dst[col] = value;
            }
        });
        return dst;
    }

    @Override
    public double get(@Nonnegative final int row, @Nonnegative final int col,
            final double defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        final int index = getIndex(row, col);
        if (index < 0) {
            return defaultValue;
        }
        return values[index];
    }

    @Override
    public double getAndSet(@Nonnegative final int row, @Nonnegative final int col,
            final double value) {
        checkIndex(row, col, numRows, numColumns);

        final int index = getIndex(row, col);
        if (index < 0) {
            throw new UnsupportedOperationException("Cannot update value in row " + row + ", col "
                    + col);
        }

        double old = values[index];
        values[index] = value;
        return old;
    }

    @Override
    public void set(@Nonnegative final int row, @Nonnegative final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);

        final int index = getIndex(row, col);
        if (index < 0) {
            throw new UnsupportedOperationException("Cannot update value in row " + row + ", col "
                    + col);
        }
        values[index] = value;
    }

    private int getIndex(@Nonnegative final int row, @Nonnegative final int col) {
        int leftIn = rowPointers[row];
        int rightEx = rowPointers[row + 1];
        final int index = Arrays.binarySearch(columnIndices, leftIn, rightEx, col);
        if (index >= 0 && index >= values.length) {
            throw new IndexOutOfBoundsException("Value index " + index + " out of range "
                    + values.length);
        }
        return index;
    }

    @Override
    public void swap(int row1, int row2) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void eachInRow(@Nonnegative final int row, @Nonnull final VectorProcedure procedure,
            final boolean nullOutput) {
        checkRowIndex(row, numRows);

        final int startIn = rowPointers[row];
        final int endEx = rowPointers[row + 1];

        if (nullOutput) {
            for (int col = 0, j = startIn; col < numColumns; col++) {
                if (j < endEx && col == columnIndices[j]) {
                    double v = values[j++];
                    procedure.apply(col, v);
                } else {
                    procedure.apply(col, 0.d);
                }
            }
        } else {
            for (int i = startIn; i < endEx; i++) {
                procedure.apply(columnIndices[i], values[i]);
            }
        }
    }

    @Override
    public void eachNonZeroInRow(@Nonnegative final int row,
            @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        final int startIn = rowPointers[row];
        final int endEx = rowPointers[row + 1];
        for (int i = startIn; i < endEx; i++) {
            int col = columnIndices[i];
            final double v = values[i];
            if (v != 0.d) {
                procedure.apply(col, v);
            }
        }
    }

    @Override
    public void eachColumnIndexInRow(@Nonnegative final int row,
            @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        final int startIn = rowPointers[row];
        final int endEx = rowPointers[row + 1];

        for (int i = startIn; i < endEx; i++) {
            procedure.apply(columnIndices[i]);
        }
    }

    @Nonnull
    public CSCMatrix toColumnMajorMatrix() {
        final int[] columnPointers = new int[numColumns + 1];
        final int[] rowIndicies = new int[nnz];
        final double[] cscValues = new double[nnz];

        // compute nnz per for each column
        for (int j = 0; j < columnIndices.length; j++) {
            columnPointers[columnIndices[j]]++;
        }
        for (int j = 0, sum = 0; j < numColumns; j++) {
            int curr = columnPointers[j];
            columnPointers[j] = sum;
            sum += curr;
        }
        columnPointers[numColumns] = nnz;

        for (int i = 0; i < numRows; i++) {
            for (int j = rowPointers[i], last = rowPointers[i + 1]; j < last; j++) {
                int col = columnIndices[j];
                int dst = columnPointers[col];

                rowIndicies[dst] = i;
                cscValues[dst] = values[j];

                columnPointers[col]++;
            }
        }

        // shift column pointers
        for (int j = 0, last = 0; j <= numColumns; j++) {
            int tmp = columnPointers[j];
            columnPointers[j] = last;
            last = tmp;
        }

        return new CSCMatrix(columnPointers, rowIndicies, cscValues, numRows, numColumns);
    }

    @Override
    public CSRMatrixBuilder builder() {
        return new CSRMatrixBuilder(values.length);
    }

}
