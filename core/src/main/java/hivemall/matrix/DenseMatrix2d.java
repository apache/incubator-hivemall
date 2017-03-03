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

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Fixed-size Dense 2-d double Matrix.
 */
public final class DenseMatrix2d extends AbstractMatrix {

    @Nonnull
    private final double[][] data;

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;

    public DenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numColumns) {
        this.data = data;
        this.numRows = data.length;
        this.numColumns = numColumns;
    }

    @Override
    public boolean readOnly() {
        return true;
    }

    @Override
    public boolean swappable() {
        return true;
    }

    @Override
    public void setDefaultValue(double value) {
        throw new UnsupportedOperationException("The defaultValue of a DenseMatrix is fixed to 0.d");
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

        return data[row].length;
    }

    @Override
    public double[] getRow(@Nonnegative final int index) {
        checkRowIndex(index, numRows);

        final double[] row = data[index];
        if (row.length == numRows) {
            return row;
        }

        final double[] result = new double[numRows];
        System.arraycopy(row, 0, result, 0, row.length);
        return result;
    }

    @Override
    public double[] getRow(@Nonnull final int index, @Nonnull final double[] dst) {
        checkRowIndex(index, numRows);

        final double[] row = data[index];
        System.arraycopy(row, 0, dst, 0, row.length);
        if (dst.length > row.length) {// zerofill
            Arrays.fill(dst, row.length, dst.length, 0.d);
        }
        return dst;
    }

    @Override
    public double get(@Nonnegative final int row, @Nonnegative final int col,
            final double defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        final double[] rowData = data[row];
        if (col >= rowData.length) {
            return defaultValue;
        }
        return rowData[col];
    }

    @Override
    public double getAndSet(@Nonnegative final int row, @Nonnegative final int col,
            final double value) {
        checkIndex(row, col, numRows, numColumns);

        final double[] rowData = data[row];
        checkColIndex(col, rowData.length);

        double old = rowData[col];
        rowData[col] = value;
        return old;
    }

    @Override
    public void set(@Nonnegative final int row, @Nonnegative final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);

        final double[] rowData = data[row];
        checkColIndex(col, rowData.length);

        rowData[col] = value;
    }

    @Override
    public void swap(final int row1, final int row2) {
        checkRowIndex(row1, numRows);
        checkRowIndex(row2, numRows);

        double[] oldRow1 = data[row1];
        data[row1] = data[row2];
        data[row2] = oldRow1;
    }

    @Override
    public void eachInRow(@Nonnegative final int row, @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        final double[] rowData = data[row];
        int col = 0;
        for (int len = rowData.length; col < len; col++) {
            procedure.apply(col, rowData[col]);
        }
        for (; col < numColumns; col++) {
            procedure.apply(col, 0.d);
        }
    }

    @Override
    public void eachNonZeroInRow(@Nonnegative final int row,
            @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        final double[] rowData = data[row];
        for (int col = 0, len = rowData.length; col < len; col++) {
            final double v = rowData[col];
            if (v == 0.d) {
                continue;
            }
            procedure.apply(col, v);
        }
    }

    @Override
    public MatrixBuilder builder() {
        return new DenseMatrixBuilder(numRows, true);
    }

}
