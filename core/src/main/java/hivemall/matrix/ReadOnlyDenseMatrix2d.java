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

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class ReadOnlyDenseMatrix2d extends Matrix {

    @Nonnull
    private final double[][] data;

    @Nonnegative
    private final int numRows;
    @Nonnegative
    private final int numColumns;

    public ReadOnlyDenseMatrix2d(@Nonnull double[][] data, @Nonnegative int numColumns) {
        this.data = data;
        this.numRows = data.length;
        this.numColumns = numColumns;
    }

    @Override
    public boolean readOnly() {
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

}
