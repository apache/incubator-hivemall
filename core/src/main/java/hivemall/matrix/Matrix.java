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

public abstract class Matrix {

    private double defaultValue;

    public Matrix() {
        this.defaultValue = 0.d;
    }

    public abstract boolean readOnly();

    public void setDefaultValue(double value) {
        this.defaultValue = value;
    }

    @Nonnegative
    public abstract int numRows();

    @Nonnegative
    public abstract int numColumns();

    @Nonnegative
    public abstract int numColumns(@Nonnegative int row);

    /**
     * @throws IndexOutOfBoundsException
     */
    public final double get(@Nonnegative final int row, @Nonnegative final int col) {
        return get(row, col, defaultValue);
    }

    /**
     * @throws IndexOutOfBoundsException
     */
    public abstract double get(@Nonnegative int row, @Nonnegative int col, double defaultValue);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public abstract void set(@Nonnegative int row, @Nonnegative int col, double value);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public abstract double getAndSet(@Nonnegative int row, @Nonnegative final int col, double value);

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
