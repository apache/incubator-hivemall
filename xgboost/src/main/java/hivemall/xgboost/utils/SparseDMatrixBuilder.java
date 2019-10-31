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
package hivemall.xgboost.utils;

import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.collections.lists.LongArrayList;
import matrix4j.utils.collections.lists.IntArrayList;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public class SparseDMatrixBuilder {

    @Nonnull
    private final LongArrayList rowPointers;
    @Nonnull
    private final IntArrayList columnIndices;
    @Nonnull
    private final FloatArrayList values;

    private int maxNumColumns;

    public SparseDMatrixBuilder(@Nonnegative int initSize) {
        this.rowPointers = new LongArrayList(initSize + 1);
        rowPointers.add(0);
        this.columnIndices = new IntArrayList(initSize);
        this.values = new FloatArrayList(initSize);
        this.maxNumColumns = 0;
    }

    public SparseDMatrixBuilder nextRow() {
        int ptr = values.size();
        rowPointers.add(ptr);
        return this;
    }

    private static final void checkColIndex(final int col) {
        if (col < 0) {
            throw new IllegalArgumentException("Found negative column index: " + col);
        }
    }

    public SparseDMatrixBuilder nextColumn(@Nonnegative int col, float value) {
        checkColIndex(col);

        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        if (value == 0.d) {
            return this;
        }

        columnIndices.add(col);
        values.add(value);
        return this;
    }

    @Nonnull
    public DMatrix buildMatrix() throws XGBoostError {
        return new DMatrix(rowPointers.toArray(true), columnIndices.toArray(true),
            values.toArray(true), DMatrix.SparseType.CSR, maxNumColumns);
    }
}
