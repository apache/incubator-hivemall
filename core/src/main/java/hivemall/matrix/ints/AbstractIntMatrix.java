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

import javax.annotation.Nonnegative;

public abstract class AbstractIntMatrix implements IntMatrix {

    protected int defaultValue;

    public AbstractIntMatrix() {
        this.defaultValue = 0;
    }

    @Override
    public void setDefaultValue(int value) {
        this.defaultValue = value;
    }

    @Override
    public int[] row() {
        int size = numRows();
        return new int[size];
    }

    @Override
    public final int get(@Nonnegative final int row, @Nonnegative final int col) {
        return get(row, col, defaultValue);
    }

    @Override
    public void incr(@Nonnegative final int row, @Nonnegative final int col) {
        incr(row, col, 1);
    }

    protected static final void checkRowIndex(final int row, final int numRows) {
        if (row < 0 || row >= numRows) {
            throw new IndexOutOfBoundsException("Row index " + row + " out of bounds " + numRows);
        }
    }

    protected static final void checkColIndex(final int col, final int numColumns) {
        if (col < 0 || col >= numColumns) {
            throw new IndexOutOfBoundsException("Col index " + col + " out of bounds " + numColumns);
        }
    }

    protected static final void checkIndex(final int row, final int col) {
        if (row < 0) {
            throw new IllegalArgumentException("Invalid row index: " + row);
        }
        if (col < 0) {
            throw new IllegalArgumentException("Invalid col index: " + col);
        }
    }

    protected static final void checkIndex(final int row, final int col, final int numRows,
            final int numColumns) {
        if (row < 0 || row >= numRows) {
            throw new IndexOutOfBoundsException("Row index " + row + " out of bounds " + numRows);
        }
        if (col < 0 || col >= numColumns) {
            throw new IndexOutOfBoundsException("Col index " + col + " out of bounds " + numColumns);
        }
    }

}
