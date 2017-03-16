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
package hivemall.matrix.dense;

import hivemall.matrix.ColumnMajorMatrix;
import hivemall.matrix.VectorProcedure;
import hivemall.matrix.builders.ColumnMajorDenseMatrixBuilder;
import hivemall.utils.lang.Preconditions;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Fixed-size Dense 2-d double Matrix.
 */
public final class ColumnMajorDenseMatrix2d extends ColumnMajorMatrix {

    @Nonnull
    private final double[][] data; // col-row

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;
    @Nonnegative
    private int nnz;

    public ColumnMajorDenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numRows) {
        this(data, numRows, nnz(data));
    }

    public ColumnMajorDenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numRows,
            @Nonnegative int nnz) {
        super();
        this.data = data;
        this.numRows = numRows;
        this.numColumns = data.length;
        this.nnz = nnz;
    }

    @Override
    public boolean isSparse() {
        return false;
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

        int numColumns = 0;
        for (int j = 0; j < data.length; j++) {
            final double[] col = data[j];
            if (col == null) {
                continue;
            }
            if (row < col.length && col[row] != 0.d) {
                numColumns++;
            }
        }
        return numColumns;
    }

    @Override
    public double[] getRow(final int index) {
        checkRowIndex(index, numRows);

        double[] row = new double[numColumns];
        return getRow(index, row);
    }

    @Override
    public double[] getRow(final int index, @Nonnull final double[] dst) {
        checkRowIndex(index, numRows);

        for (int j = 0; j < data.length; j++) {
            final double[] col = data[j];
            if (col == null) {
                continue;
            }
            if (index < col.length) {
                dst[j] = col[index];
            }
        }
        return dst;
    }

    @Override
    public double get(final int row, final int col, final double defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        final double[] colData = data[col];
        if (colData == null || row >= colData.length) {
            return defaultValue;
        }
        return colData[row];
    }

    @Override
    public double getAndSet(final int row, final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);

        final double[] colData = data[col];
        Preconditions.checkNotNull(colData, "col does not exists: " + col);
        checkRowIndex(row, colData.length);

        final double old = colData[row];
        colData[row] = value;
        if (old == 0.d && value != 0.d) {
            ++nnz;
        }
        return old;
    }

    @Override
    public void set(final int row, final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);
        if (value == 0.d) {
            return;
        }

        final double[] colData = data[col];
        Preconditions.checkNotNull(colData, "col does not exists: " + col);
        checkRowIndex(row, colData.length);

        if (colData[row] == 0.d) {
            ++nnz;
        }
        colData[row] = value;
    }

    @Override
    public void swap(int row1, int row2) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void eachInColumn(final int col, @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        final double[] colData = data[col];
        if (colData == null) {
            return;
        }
        int row = 0;
        for (int len = colData.length; row < len; row++) {
            procedure.apply(row, colData[row]);
        }
        for (; row < numRows; row++) {
            procedure.apply(row, 0.d);
        }
    }

    @Override
    public void eachInNonZeroColumn(final int col, @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        final double[] colData = data[col];
        if (colData == null) {
            return;
        }
        int row = 0;
        for (int len = colData.length; row < len; row++) {
            final double v = colData[row];
            if (v == 0.d) {
                continue;
            }
            procedure.apply(row, v);
        }
    }

    @Override
    public RowMajorDenseMatrix2d toRowMajorMatrix() {
        final double[][] rowcol = new double[numRows][numColumns];
        int nnz = 0;
        for (int j = 0; j < data.length; j++) {
            final double[] colData = data[j];
            if (colData == null) {
                continue;
            }
            for (int i = 0; i < colData.length; i++) {
                final double v = colData[i];
                if (v == 0.d) {
                    continue;
                }
                rowcol[i][j] = v;
                nnz++;
            }
        }
        for (int i = 0; i < rowcol.length; i++) {
            final double[] row = rowcol[i];
            final int last = numColumns - 1;
            int maxj = last;
            for (; maxj >= 0; maxj--) {
                if (row[maxj] != 0.d) {
                    break;
                }
            }
            if (maxj == last) {
                continue;
            } else if (maxj < 0) {
                rowcol[i] = null;
                continue;
            }
            final double[] dstRow = new double[maxj + 1];
            System.arraycopy(row, 0, dstRow, 0, dstRow.length);
            rowcol[i] = dstRow;
        }

        return new RowMajorDenseMatrix2d(rowcol, numColumns, nnz);
    }

    @Override
    public ColumnMajorDenseMatrixBuilder builder() {
        return new ColumnMajorDenseMatrixBuilder(numColumns);
    }

    private static int nnz(@Nonnull final double[][] data) {
        int count = 0;
        for (int j = 0; j < data.length; j++) {
            final double[] col = data[j];
            if (col == null) {
                continue;
            }
            for (int i = 0; i < col.length; i++) {
                if (col[i] != 0.d) {
                    ++count;
                }
            }
        }
        return count;
    }

}
