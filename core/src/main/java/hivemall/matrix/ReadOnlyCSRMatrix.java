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

import hivemall.utils.lang.Preconditions;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Read-only CSR Matrix.
 * 
 * @see http://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
 */
public final class ReadOnlyCSRMatrix extends Matrix {

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

    public ReadOnlyCSRMatrix(@Nonnull int[] rowPointers, @Nonnull int[] columnIndices,
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
    public int numColumns(@Nonnegative final int row) {
        checkRowIndex(row, numRows);

        int columns = rowPointers[row + 1] - rowPointers[row];
        return columns;
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

}
