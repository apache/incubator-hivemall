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
package hivemall.matrix.ints;

import hivemall.vector.VectorProcedure;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class ColumnMajorDenseIntMatrix2d extends ColumnMajorIntMatrix {

    @Nonnull
    private final int[][] data; // col-row

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;

    public ColumnMajorDenseIntMatrix2d(@Nonnull int[][] data, @Nonnegative int numRows) {
        super();
        this.data = data;
        this.numRows = numRows;
        this.numColumns = data.length;
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
    public int numRows() {
        return numRows;
    }

    @Override
    public int numColumns() {
        return numColumns;
    }

    @Override
    public int[] getRow(final int index) {
        checkRowIndex(index, numRows);

        int[] row = new int[numColumns];
        return getRow(index, row);
    }

    @Override
    public int[] getRow(final int index, @Nonnull final int[] dst) {
        checkRowIndex(index, numRows);

        for (int j = 0; j < data.length; j++) {
            final int[] col = data[j];
            if (index < col.length) {
                dst[j] = col[index];
            }
        }
        return dst;
    }

    @Override
    public int get(final int row, final int col, final int defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        final int[] colData = data[col];
        if (row >= colData.length) {
            return defaultValue;
        }
        return colData[row];
    }

    @Override
    public int getAndSet(final int row, final int col, final int value) {
        checkIndex(row, col, numRows, numColumns);

        final int[] colData = data[col];
        checkRowIndex(row, colData.length);

        final int old = colData[row];
        colData[row] = value;
        return old;
    }

    @Override
    public void set(final int row, final int col, final int value) {
        checkIndex(row, col, numRows, numColumns);
        if (value == 0) {
            return;
        }

        final int[] colData = data[col];
        checkRowIndex(row, colData.length);
        colData[row] = value;
    }

    @Override
    public void incr(final int row, final int col, final int delta) {
        checkIndex(row, col, numRows, numColumns);

        final int[] colData = data[col];
        checkRowIndex(row, colData.length);

        colData[row] += delta;
    }

    @Override
    public void eachInColumn(final int col, @Nonnull final VectorProcedure procedure,
            final boolean nullOutput) {
        checkColIndex(col, numColumns);

        final int[] colData = data[col];
        if (colData == null) {
            return;
        }
        int row = 0;
        for (int len = colData.length; row < len; row++) {
            procedure.apply(row, colData[row]);
        }
        if (nullOutput) {
            for (; row < numRows; row++) {
                procedure.apply(row, defaultValue);
            }
        }
    }

    @Override
    public void eachInNonZeroColumn(final int col, @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        final int[] colData = data[col];
        if (colData == null) {
            return;
        }
        int row = 0;
        for (int len = colData.length; row < len; row++) {
            final int v = colData[row];
            if (v == 0) {
                continue;
            }
            procedure.apply(row, v);
        }
    }

}
