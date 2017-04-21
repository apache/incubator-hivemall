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

import hivemall.math.matrix.ColumnMajorMatrix;
import hivemall.math.matrix.builders.CSCMatrixBuilder;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.Preconditions;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * @link http://netlib.org/linalg/html_templates/node92.html#SECTION00931200000000000000
 */
public final class CSCMatrix extends ColumnMajorMatrix {

    @Nonnull
    private final int[] columnPointers;
    @Nonnull
    private final int[] rowIndicies;
    @Nonnull
    private final double[] values;

    private final int numRows;
    private final int numColumns;
    private final int nnz;

    public CSCMatrix(@Nonnull int[] columnPointers, @Nonnull int[] rowIndicies,
            @Nonnull double[] values, int numRows, int numColumns) {
        super();
        Preconditions.checkArgument(columnPointers.length >= 1,
            "rowPointers must be greather than 0: " + columnPointers.length);
        Preconditions.checkArgument(rowIndicies.length == values.length, "#rowIndicies ("
                + rowIndicies.length + ") must be equals to #values (" + values.length + ")");
        this.columnPointers = columnPointers;
        this.rowIndicies = rowIndicies;
        this.values = values;
        this.numRows = numRows;
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
    public int numColumns(final int row) {
        checkRowIndex(row, numRows);

        return ArrayUtils.count(rowIndicies, row);
    }

    @Override
    public double[] getRow(int index) {
        checkRowIndex(index, numRows);

        final double[] row = new double[numColumns];

        final int numCols = columnPointers.length - 1;
        for (int j = 0; j < numCols; j++) {
            final int k = Arrays.binarySearch(rowIndicies, columnPointers[j],
                columnPointers[j + 1], index);
            if (k >= 0) {
                row[j] = values[k];
            }
        }

        return row;
    }

    @Override
    public double[] getRow(final int index, @Nonnull final double[] dst) {
        checkRowIndex(index, numRows);

        final int last = Math.min(dst.length, columnPointers.length - 1);
        for (int j = 0; j < last; j++) {
            final int k = Arrays.binarySearch(rowIndicies, columnPointers[j],
                columnPointers[j + 1], index);
            if (k >= 0) {
                dst[j] = values[k];
            }
        }

        return dst;
    }

    @Override
    public void getRow(final int index, @Nonnull final Vector row) {
        checkRowIndex(index, numRows);
        row.clear();

        for (int j = 0, last = columnPointers.length - 1; j < last; j++) {
            final int k = Arrays.binarySearch(rowIndicies, columnPointers[j],
                columnPointers[j + 1], index);
            if (k >= 0) {
                double v = values[k];
                row.set(j, v);
            }
        }
    }

    @Override
    public double get(final int row, final int col, final double defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        int index = getIndex(row, col);
        if (index < 0) {
            return defaultValue;
        }
        return values[index];
    }

    @Override
    public double getAndSet(final int row, final int col, final double value) {
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
    public void set(final int row, final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);

        final int index = getIndex(row, col);
        if (index < 0) {
            throw new UnsupportedOperationException("Cannot update value in row " + row + ", col "
                    + col);
        }
        values[index] = value;
    }

    private int getIndex(@Nonnegative final int row, @Nonnegative final int col) {
        int leftIn = columnPointers[col];
        int rightEx = columnPointers[col + 1];
        final int index = Arrays.binarySearch(rowIndicies, leftIn, rightEx, row);
        if (index >= 0 && index >= values.length) {
            throw new IndexOutOfBoundsException("Value index " + index + " out of range "
                    + values.length);
        }
        return index;
    }

    @Override
    public void swap(final int row1, final int row2) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void eachInColumn(final int col, @Nonnull final VectorProcedure procedure,
            final boolean nullOutput) {
        checkColIndex(col, numColumns);

        final int startIn = columnPointers[col];
        final int endEx = columnPointers[col + 1];

        if (nullOutput) {
            for (int row = 0, i = startIn; row < numRows; row++) {
                if (i < endEx && row == rowIndicies[i]) {
                    double v = values[i++];
                    procedure.apply(row, v);
                } else {
                    procedure.apply(row, 0.d);
                }
            }
        } else {
            for (int j = startIn; j < endEx; j++) {
                int row = rowIndicies[j];
                double v = values[j];
                procedure.apply(row, v);
            }
        }
    }

    @Override
    public void eachNonZeroInColumn(final int col, @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        final int startIn = columnPointers[col];
        final int endEx = columnPointers[col + 1];
        for (int j = startIn; j < endEx; j++) {
            int row = rowIndicies[j];
            final double v = values[j];
            if (v != 0.d) {
                procedure.apply(row, v);
            }
        }
    }

    @Override
    public CSRMatrix toRowMajorMatrix() {
        final int[] rowPointers = new int[numRows + 1];
        final int[] colIndicies = new int[nnz];
        final double[] csrValues = new double[nnz];

        // compute nnz per for each row
        for (int i = 0; i < rowIndicies.length; i++) {
            rowPointers[rowIndicies[i]]++;
        }
        for (int i = 0, sum = 0; i < numRows; i++) {
            int curr = rowPointers[i];
            rowPointers[i] = sum;
            sum += curr;
        }
        rowPointers[numRows] = nnz;

        for (int j = 0; j < numColumns; j++) {
            for (int i = columnPointers[j], last = columnPointers[j + 1]; i < last; i++) {
                int col = rowIndicies[i];
                int dst = rowPointers[col];

                colIndicies[dst] = j;
                csrValues[dst] = values[i];

                rowPointers[col]++;
            }
        }

        // shift column pointers
        for (int i = 0, last = 0; i <= numRows; i++) {
            int tmp = rowPointers[i];
            rowPointers[i] = last;
            last = tmp;
        }

        return new CSRMatrix(rowPointers, colIndicies, csrValues, numColumns);
    }

    @Override
    public CSCMatrixBuilder builder() {
        return new CSCMatrixBuilder(nnz);
    }

}
