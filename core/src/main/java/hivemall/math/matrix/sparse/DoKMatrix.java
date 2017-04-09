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
package hivemall.math.matrix.sparse;

import hivemall.annotations.Experimental;
import hivemall.math.matrix.AbstractMatrix;
import hivemall.math.matrix.ColumnMajorMatrix;
import hivemall.math.matrix.RowMajorMatrix;
import hivemall.math.matrix.builders.DoKMatrixBuilder;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.collections.maps.Long2DoubleOpenHashTable;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

@Experimental
public final class DoKMatrix extends AbstractMatrix {

    @Nonnull
    private final Long2DoubleOpenHashTable elements;
    @Nonnegative
    private int numRows;
    @Nonnegative
    private int numColumns;
    @Nonnegative
    private int nnz;

    public DoKMatrix() {
        this(0, 0);
    }

    public DoKMatrix(@Nonnegative int numRows, @Nonnegative int numCols) {
        this(numRows, numCols, 0.05f);
    }

    public DoKMatrix(@Nonnegative int numRows, @Nonnegative int numCols, @Nonnegative float sparsity) {
        super();
        Preconditions.checkArgument(sparsity >= 0.f && sparsity <= 1.f, "Invalid Sparsity value: "
                + sparsity);
        int initialCapacity = Math.max(16384, Math.round(numRows * numCols * sparsity));
        this.elements = new Long2DoubleOpenHashTable(initialCapacity);
        elements.defaultReturnValue(0.d);
        this.numRows = numRows;
        this.numColumns = numCols;
        this.nnz = 0;
    }

    public DoKMatrix(@Nonnegative int initSize) {
        super();
        int initialCapacity = Math.max(initSize, 16384);
        this.elements = new Long2DoubleOpenHashTable(initialCapacity);
        elements.defaultReturnValue(0.d);
        this.numRows = 0;
        this.numColumns = 0;
        this.nnz = 0;
    }

    @Override
    public boolean isSparse() {
        return true;
    }

    @Override
    public boolean isRowMajorMatrix() {
        return false;
    }

    @Override
    public boolean isColumnMajorMatrix() {
        return false;
    }

    @Override
    public boolean readOnly() {
        return false;
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
        int count = 0;
        for (int j = 0; j < numColumns; j++) {
            long index = index(row, j);
            if (elements.containsKey(index)) {
                count++;
            }
        }
        return count;
    }

    @Override
    public double[] getRow(@Nonnegative final int index) {
        double[] dst = row();
        return getRow(index, dst);
    }

    @Override
    public double[] getRow(@Nonnegative final int row, @Nonnull final double[] dst) {
        checkRowIndex(row, numRows);

        final int end = Math.min(dst.length, numColumns);
        for (int col = 0; col < end; col++) {
            long k = index(row, col);
            double v = elements.get(k);
            dst[col] = v;
        }

        return dst;
    }

    @Override
    public void getRow(@Nonnegative final int index, @Nonnull final Vector row) {
        checkRowIndex(index, numRows);
        row.clear();

        for (int col = 0; col < numColumns; col++) {
            long k = index(index, col);
            final double v = elements.get(k, 0.d);
            if (v != 0.d) {
                row.set(col, v);
            }
        }
    }

    @Override
    public double get(@Nonnegative final int row, @Nonnegative final int col,
            final double defaultValue) {
        checkIndex(row, col, numRows, numColumns);

        long index = index(row, col);
        return elements.get(index, defaultValue);
    }

    @Override
    public void set(@Nonnegative final int row, @Nonnegative final int col, final double value) {
        checkIndex(row, col);

        if (value == 0.d) {
            return;
        }

        long index = index(row, col);
        if (elements.put(index, value, 0.d) == 0.d) {
            nnz++;
            this.numRows = Math.max(numRows, row + 1);
            this.numColumns = Math.max(numColumns, col + 1);
        }
    }

    @Override
    public double getAndSet(@Nonnegative final int row, @Nonnegative final int col,
            final double value) {
        checkIndex(row, col);

        long index = index(row, col);
        double old = elements.put(index, value, 0.d);
        if (old == 0.d) {
            nnz++;
            this.numRows = Math.max(numRows, row + 1);
            this.numColumns = Math.max(numColumns, col + 1);
        }
        return old;
    }

    @Override
    public void swap(@Nonnegative final int row1, @Nonnegative final int row2) {
        checkRowIndex(row1, numRows);
        checkRowIndex(row2, numRows);

        for (int j = 0; j < numColumns; j++) {
            final long i1 = index(row1, j);
            final long i2 = index(row2, j);

            final int k1 = elements._findKey(i1);
            final int k2 = elements._findKey(i2);

            if (k1 >= 0) {
                if (k2 >= 0) {
                    double v1 = elements._get(k1);
                    double v2 = elements._set(k2, v1);
                    elements._set(k1, v2);
                } else {// k1>=0 and k2<0
                    double v1 = elements._remove(k1);
                    elements.put(i2, v1);
                }
            } else if (k2 >= 0) {// k2>=0 and k1 < 0
                double v2 = elements._remove(k2);
                elements.put(i1, v2);
            } else {//k1<0 and k2<0
                continue;
            }
        }
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
                    procedure.apply(col, 0.d);
                }
            } else {
                double v = elements._get(key);
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
            final double v = elements.get(i, 0.d);
            if (v != 0.d) {
                procedure.apply(col, v);
            }
        }
    }

    @Override
    public void eachColumnIndexInRow(int row, VectorProcedure procedure) {
        checkRowIndex(row, numRows);

        for (int col = 0; col < numColumns; col++) {
            long i = index(row, col);
            final int key = elements._findKey(i);
            if (key != -1) {
                procedure.apply(col);
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
                    procedure.apply(row, 0.d);
                }
            } else {
                double v = elements._get(key);
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
            final double v = elements.get(i, 0.d);
            if (v != 0.d) {
                procedure.apply(row, v);
            }
        }
    }

    @Override
    public RowMajorMatrix toRowMajorMatrix() {
        throw new UnsupportedOperationException("Not yet supported");
    }

    @Override
    public ColumnMajorMatrix toColumnMajorMatrix() {
        throw new UnsupportedOperationException("Not yet supported");
    }

    @Override
    public DoKMatrixBuilder builder() {
        return new DoKMatrixBuilder(elements.size());
    }

    @Nonnegative
    private static long index(@Nonnegative final int row, @Nonnegative final int col) {
        return Primitives.toLong(row, col);
    }

}
