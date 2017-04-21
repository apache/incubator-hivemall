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
package hivemall.math.matrix.builders;

import hivemall.math.matrix.dense.ColumnMajorDenseMatrix2d;
import hivemall.utils.collections.arrays.SparseDoubleArray;
import hivemall.utils.collections.maps.IntOpenHashTable;
import hivemall.utils.collections.maps.IntOpenHashTable.IMapIterator;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class ColumnMajorDenseMatrixBuilder extends MatrixBuilder {

    @Nonnull
    private final IntOpenHashTable<SparseDoubleArray> col2rows;
    private int row;
    private int maxNumColumns;
    private int nnz;

    public ColumnMajorDenseMatrixBuilder(int initSize) {
        this.col2rows = new IntOpenHashTable<SparseDoubleArray>(initSize);
        this.row = 0;
        this.maxNumColumns = 0;
        this.nnz = 0;
    }

    @Override
    public ColumnMajorDenseMatrixBuilder nextRow() {
        row++;
        return this;
    }

    @Override
    public ColumnMajorDenseMatrixBuilder nextColumn(@Nonnegative final int col, final double value) {
        if (value == 0.d) {
            return this;
        }

        SparseDoubleArray rows = col2rows.get(col);
        if (rows == null) {
            rows = new SparseDoubleArray(4);
            col2rows.put(col, rows);
        }
        rows.put(row, value);
        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        nnz++;
        return this;
    }

    @Override
    public ColumnMajorDenseMatrix2d buildMatrix() {
        final double[][] data = new double[maxNumColumns][];

        final IMapIterator<SparseDoubleArray> itor = col2rows.entries();
        while (itor.next() != -1) {
            int col = itor.getKey();
            SparseDoubleArray rows = itor.getValue();
            data[col] = rows.toArray();
        }

        return new ColumnMajorDenseMatrix2d(data, row, nnz);
    }

}
