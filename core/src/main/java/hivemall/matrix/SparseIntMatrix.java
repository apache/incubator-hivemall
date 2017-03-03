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

import hivemall.utils.collections.Long2IntOpenHashTable;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Fixed-size Sparse Int Matrix.
 */
public final class SparseIntMatrix extends AbstractIntMatrix {

    @Nonnull
    private final Long2IntOpenHashTable elements;
    @Nonnegative
    private int numRows;
    @Nonnegative
    private int numColumns;

    public SparseIntMatrix() {
        this(0, 0);
    }

    public SparseIntMatrix(@Nonnegative int numRows, @Nonnegative int numCols) {
        this(numCols, numCols, 0.05f);
    }

    public SparseIntMatrix(@Nonnegative int numRows, @Nonnegative int numCols,
            @Nonnegative float sparsity) {
        Preconditions.checkArgument(sparsity >= 0.f && sparsity <= 1.f, "Invalid Sparsity value: "
                + sparsity);
        int initialCapacity = Math.max(16384, Math.round(numRows * numCols * sparsity));
        this.elements = new Long2IntOpenHashTable(initialCapacity);
        this.numRows = numRows;
        this.numColumns = numCols;
    }

    @Override
    public boolean readOnly() {
        return false;
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
    public int[] getRow(@Nonnegative final int index) {
        int[] dst = row();
        return getRow(index, dst);
    }

    @Override
    public int[] getRow(@Nonnegative final int row, @Nonnull final int[] dst) {
        checkRowIndex(row, numRows);

        final int end = Math.min(dst.length, numColumns);
        for (int col = 0; col < end; col++) {
            long index = index(row, col);
            int v = elements.get(index, defaultValue);
            dst[col] = v;
        }

        return dst;
    }

    @Override
    public int get(@Nonnegative final int row, @Nonnegative final int col, final int defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        long index = index(row, col);
        return elements.get(index, defaultValue);
    }

    @Override
    public void set(@Nonnegative final int row, @Nonnegative final int col, final int value) {
        checkIndex(row, col);

        long index = index(row, col);
        elements.put(index, value);
        this.numRows = Math.max(numRows, row + 1);
        this.numColumns = Math.max(numColumns, col + 1);
    }

    @Override
    public int getAndSet(@Nonnegative final int row, @Nonnegative final int col, final int value) {
        checkIndex(row, col);

        long index = index(row, col);
        int old = elements.put(index, value);
        this.numRows = Math.max(numRows, row + 1);
        this.numColumns = Math.max(numColumns, col + 1);
        return old;
    }

    @Override
    public void incr(@Nonnegative final int row, @Nonnegative final int col, final int delta) {
        checkIndex(row, col);

        long index = index(row, col);
        elements.incr(index, delta);
        this.numRows = Math.max(numRows, row + 1);
        this.numColumns = Math.max(numColumns, col + 1);
    }

    @Override
    public void eachInRow(@Nonnegative final int row, @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        for (int col = 0; col < numColumns; col++) {
            long i = index(row, col);
            int v = elements.get(i, defaultValue);
            procedure.apply(col, v);
        }
    }

    @Override
    public void eachNonZeroInRow(@Nonnegative final int row,
            @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        for (int col = 0; col < numColumns; col++) {
            long i = index(row, col);
            int v = elements.get(i, 0);
            if (v != 0) {
                procedure.apply(col, v);
            }
        }
    }

    @Nonnegative
    private static long index(@Nonnegative final int row, @Nonnegative final int col) {
        return Primitives.toLong(row, col);
    }
}
