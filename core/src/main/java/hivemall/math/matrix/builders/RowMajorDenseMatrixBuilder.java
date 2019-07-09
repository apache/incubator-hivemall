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

import hivemall.math.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.utils.collections.arrays.SparseDoubleArray;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class RowMajorDenseMatrixBuilder extends MatrixBuilder {

    @Nonnull
    private final List<double[]> rows;
    private int maxNumColumns;
    private int nnz;

    @Nonnull
    private final SparseDoubleArray rowProbe;

    public RowMajorDenseMatrixBuilder(@Nonnegative int initSize) {
        super();
        this.rows = new ArrayList<double[]>(initSize);
        this.maxNumColumns = 0;
        this.nnz = 0;
        this.rowProbe = new SparseDoubleArray(32);
    }

    @Override
    public RowMajorDenseMatrixBuilder nextColumn(@Nonnegative final int col, final double value) {
        checkColIndex(col);

        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        if (value == 0.d) {
            return this;
        }
        rowProbe.put(col, value);
        nnz++;
        return this;
    }

    @Override
    public RowMajorDenseMatrixBuilder nextRow() {
        double[] row = rowProbe.toArray();
        rowProbe.clear();
        rows.add(row);
        //this.maxNumColumns = Math.max(row.length, maxNumColumns);
        return this;
    }

    @Override
    public void nextRow(@Nonnull double[] row) {
        for (double v : row) {
            if (v != 0.d) {
                nnz++;
            }
        }
        rows.add(row);
        this.maxNumColumns = Math.max(row.length, maxNumColumns);
    }

    @Override
    public RowMajorDenseMatrix2d buildMatrix() {
        int numRows = rows.size();
        double[][] data = rows.toArray(new double[numRows][]);
        return new RowMajorDenseMatrix2d(data, maxNumColumns, nnz);
    }

}
