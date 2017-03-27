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

import hivemall.utils.collections.maps.Long2IntOpenHashTable;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;
import hivemall.vector.VectorProcedure;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Dictionary-of-Key Sparse Int Matrix.
 */
public final class DoKIntMatrix extends AbstractIntMatrix {

    @Nonnull
    private final Long2IntOpenHashTable elements;
    @Nonnegative
    private int numRows;
    @Nonnegative
    private int numColumns;

    public DoKIntMatrix() {
        this(0, 0);
    }

    public DoKIntMatrix(@Nonnegative int numRows, @Nonnegative int numCols) {
        this(numRows, numCols, 0.05f);
    }

    public DoKIntMatrix(@Nonnegative int numRows, @Nonnegative int numCols,
            @Nonnegative float sparsity) {
        Preconditions.checkArgument(sparsity >= 0.f && sparsity <= 1.f, "Invalid Sparsity value: "
                + sparsity);
        int initialCapacity = Math.max(16384, Math.round(numRows * numCols * sparsity));
        this.elements = new Long2IntOpenHashTable(initialCapacity);
        this.numRows = numRows;
        this.numColumns = numCols;
    }

    private DoKIntMatrix(@Nonnull Long2IntOpenHashTable elements, @Nonnegative int numRows,
            @Nonnegative int numColumns) {
        this.elements = elements;
        this.numRows = numRows;
        this.numColumns = numColumns;
    }

    @Override
    public boolean isSparse() {
        return true;
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
    public void eachInRow(@Nonnegative final int row, @Nonnull final VectorProcedure procedure,
            final boolean nullOutput) {
        checkRowIndex(row, numRows);

        for (int col = 0; col < numColumns; col++) {
            long i = index(row, col);
            final int key = elements._findKey(i);
            if (key < 0) {
                if (nullOutput) {
                    procedure.apply(col, defaultValue);
                }
            } else {
                int v = elements._get(key);
                procedure.apply(col, v);
            }
        }
    }

    @Override
    public void eachNonZeroInRow(@Nonnegative final int row,
            @Nonnull final VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        for (int col = 0; col < numColumns; col++) {
            long i = index(row, col);
            final int v = elements.get(i, 0);
            if (v != 0) {
                procedure.apply(col, v);
            }
        }
    }

    @Override
    public void eachInColumn(@Nonnegative final int col, @Nonnull final VectorProcedure procedure,
            final boolean nullOutput) {
        checkColIndex(col, numColumns);

        for (int row = 0; row < numRows; row++) {
            long i = index(row, col);
            final int key = elements._findKey(i);
            if (key < 0) {
                if (nullOutput) {
                    procedure.apply(row, defaultValue);
                }
            } else {
                int v = elements._get(key);
                procedure.apply(row, v);
            }
        }
    }

    @Override
    public void eachNonZeroInColumn(@Nonnegative final int col,
            @Nonnull final VectorProcedure procedure) {
        checkColIndex(col, numColumns);

        for (int row = 0; row < numRows; row++) {
            long i = index(row, col);
            final int v = elements.get(i, 0);
            if (v != 0) {
                procedure.apply(row, v);
            }
        }
    }

    @Nonnegative
    private static long index(@Nonnegative final int row, @Nonnegative final int col) {
        return Primitives.toLong(row, col);
    }

    @Nonnull
    public static DoKIntMatrix build(@Nonnull final int[][] matrix, boolean rowMajorInput,
            boolean nonZeroOnly) {
        if (rowMajorInput) {
            return buildFromRowMajorMatrix(matrix, nonZeroOnly);
        } else {
            return buildFromColumnMajorMatrix(matrix, nonZeroOnly);
        }
    }

    @Nonnull
    private static DoKIntMatrix buildFromRowMajorMatrix(@Nonnull final int[][] rowMajorMatrix,
            boolean nonZeroOnly) {
        final Long2IntOpenHashTable elements = new Long2IntOpenHashTable(rowMajorMatrix.length * 3);

        int numRows = rowMajorMatrix.length, numColumns = 0;
        for (int i = 0; i < rowMajorMatrix.length; i++) {
            final int[] row = rowMajorMatrix[i];
            if (row == null) {
                continue;
            }
            numColumns = Math.max(numColumns, row.length);
            for (int col = 0; col < row.length; col++) {
                int value = row[col];
                if (nonZeroOnly && value == 0) {
                    continue;
                }
                long index = index(i, col);
                elements.put(index, value);
            }
        }

        return new DoKIntMatrix(elements, numRows, numColumns);
    }

    @Nonnull
    private static DoKIntMatrix buildFromColumnMajorMatrix(
            @Nonnull final int[][] columnMajorMatrix, boolean nonZeroOnly) {
        final Long2IntOpenHashTable elements = new Long2IntOpenHashTable(
            columnMajorMatrix.length * 3);

        int numRows = 0, numColumns = columnMajorMatrix.length;
        for (int j = 0; j < columnMajorMatrix.length; j++) {
            final int[] col = columnMajorMatrix[j];
            if (col == null) {
                continue;
            }
            numRows = Math.max(numRows, col.length);
            for (int row = 0; row < col.length; row++) {
                int value = col[row];
                if (nonZeroOnly && value == 0) {
                    continue;
                }
                long index = index(row, j);
                elements.put(index, value);
            }
        }

        return new DoKIntMatrix(elements, numRows, numColumns);
    }

}
