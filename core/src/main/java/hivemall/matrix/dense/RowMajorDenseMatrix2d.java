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

import hivemall.matrix.RowMajorMatrix;
import hivemall.matrix.builders.RowMajorDenseMatrixBuilder;
import hivemall.utils.lang.Preconditions;
import hivemall.vector.VectorProcedure;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Fixed-size Dense 2-d double Matrix.
 */
public final class RowMajorDenseMatrix2d extends RowMajorMatrix {

    @Nonnull
    private final double[][] data;

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;
    @Nonnegative
    private int nnz;

    public RowMajorDenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numColumns) {
        this(data, numColumns, nnz(data));
    }

    public RowMajorDenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numColumns,
            @Nonnegative int nnz) {
        super();
        this.data = data;
        this.numRows = data.length;
        this.numColumns = numColumns;
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
        return true;
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

        final double[] r = data[row];
        if (r == null) {
            return 0;
        }
        return r.length;
    }

    @Override
    public double[] getRow(@Nonnegative final int index) {
        checkRowIndex(index, numRows);

        final double[] row = data[index];
        if (row == null) {
            return new double[0];
        } else if (row.length == numRows) {
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
        if (row == null) {
            return new double[0];
        }

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
        if (rowData == null || col >= rowData.length) {
            return defaultValue;
        }
        return rowData[col];
    }

    @Override
    public double getAndSet(@Nonnegative final int row, @Nonnegative final int col,
            final double value) {
        checkIndex(row, col, numRows, numColumns);

        final double[] rowData = data[row];
        Preconditions.checkNotNull(rowData, "row does not exists: " + row);
        checkColIndex(col, rowData.length);

        double old = rowData[col];
        rowData[col] = value;
        if (old == 0.d && value != 0.d) {
            ++nnz;
        }
        return old;
    }

    @Override
    public void set(@Nonnegative final int row, @Nonnegative final int col, final double value) {
        checkIndex(row, col, numRows, numColumns);
        if (value == 0.d) {
            return;
        }

        final double[] rowData = data[row];
        Preconditions.checkNotNull(rowData, "row does not exists: " + row);
        checkColIndex(col, rowData.length);

        if (rowData[col] == 0.d) {
            ++nnz;
        }
        rowData[col] = value;
    }

    @Override
    public void swap(@Nonnegative final int row1, @Nonnegative final int row2) {
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
        if (rowData == null) {
            return;
        }
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
        if (rowData == null) {
            return;
        }
        for (int col = 0, len = rowData.length; col < len; col++) {
            final double v = rowData[col];
            if (v == 0.d) {
                continue;
            }
            procedure.apply(col, v);
        }
    }

    @Override
    public void eachInColumn(@Nonnegative final int col, @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        for (int row = 0; row < numRows; row++) {
            final double[] rowData = data[row];
            if (rowData == null) {
                continue;
            }
            if (col < rowData.length) {
                procedure.apply(row, rowData[col]);
            } else {
                procedure.apply(row, 0.d);
            }
        }
    }

    @Override
    public void eachInNonZeroColumn(@Nonnegative final int col,
            @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        for (int row = 0; row < numRows; row++) {
            final double[] rowData = data[row];
            if (rowData == null) {
                continue;
            }
            if (col < rowData.length) {
                double v = rowData[col];
                if (v != 0.d) {
                    procedure.apply(row, v);
                }
            }
        }
    }

    @Override
    public ColumnMajorDenseMatrix2d toColumnMajorMatrix() {
        final double[][] colrow = new double[numColumns][numRows];
        int nnz = 0;
        for (int i = 0; i < data.length; i++) {
            final double[] rowData = data[i];
            if (rowData == null) {
                continue;
            }
            for (int j = 0; j < rowData.length; j++) {
                final double v = rowData[j];
                if (v == 0.d) {
                    continue;
                }
                colrow[j][i] = v;
                nnz++;
            }
        }
        for (int j = 0; j < colrow.length; j++) {
            final double[] col = colrow[j];
            final int last = numRows - 1;
            int maxi = last;
            for (; maxi >= 0; maxi--) {
                if (col[maxi] != 0.d) {
                    break;
                }
            }
            if (maxi == last) {
                continue;
            } else if (maxi < 0) {
                colrow[j] = null;
                continue;
            }
            final double[] dstCol = new double[maxi + 1];
            System.arraycopy(col, 0, dstCol, 0, dstCol.length);
            colrow[j] = dstCol;
        }

        return new ColumnMajorDenseMatrix2d(colrow, numRows, nnz);
    }

    @Override
    public RowMajorDenseMatrixBuilder builder() {
        return new RowMajorDenseMatrixBuilder(numRows);
    }

    private static int nnz(@Nonnull final double[][] data) {
        int count = 0;
        for (int i = 0; i < data.length; i++) {
            final double[] row = data[i];
            if (row == null) {
                continue;
            }
            for (int j = 0; j < row.length; j++) {
                if (row[j] != 0.d) {
                    ++count;
                }
            }
        }
        return count;
    }

}
