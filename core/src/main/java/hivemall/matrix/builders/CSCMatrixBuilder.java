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
package hivemall.matrix.builders;

import hivemall.matrix.sparse.CSCMatrix;
import hivemall.utils.collections.lists.DoubleArrayList;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.lang.ArrayUtils;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class CSCMatrixBuilder extends MatrixBuilder {

    @Nonnull
    private final IntArrayList rows;
    @Nonnull
    private final IntArrayList cols;
    @Nonnull
    private final DoubleArrayList values;

    private int row;
    private int maxNumColumns;

    public CSCMatrixBuilder(int initSize) {
        super();
        this.rows = new IntArrayList(initSize);
        this.cols = new IntArrayList(initSize);
        this.values = new DoubleArrayList(initSize);
        this.row = 0;
        this.maxNumColumns = 0;
    }

    @Override
    public MatrixBuilder nextRow() {
        row++;
        return this;
    }

    @Override
    public MatrixBuilder nextColumn(@Nonnegative final int col, final double value) {
        rows.add(row);
        cols.add(col);
        values.add((float) value);
        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        return this;
    }

    @Override
    public CSCMatrix buildMatrix() {
        if (rows.isEmpty() || cols.isEmpty()) {
            throw new IllegalStateException("No element in the matrix");
        }

        int[] colsArray = cols.toArray(true);
        final int[] rowsIndicies = rows.toArray(true);
        final double[] valuesArray = values.toArray(true);

        // convert to column major
        ArrayUtils.sort(colsArray, rowsIndicies, valuesArray);

        final IntArrayList colPointers = new IntArrayList(1024);
        int prev = colsArray[0];
        for (int j = 1; j < colsArray.length; j++) {
            int curr = colsArray[j];
            if (curr != prev) {
                colPointers.add(curr);
            }
        }
        colsArray = null; // help GC

        return new CSCMatrix(colPointers.toArray(true), rowsIndicies, valuesArray, row,
            maxNumColumns);
    }

}
