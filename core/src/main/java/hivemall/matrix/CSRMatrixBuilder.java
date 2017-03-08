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

import hivemall.utils.collections.lists.DoubleArrayList;
import hivemall.utils.collections.lists.IntArrayList;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Compressed Sparse Row Matrix builder.
 */
public final class CSRMatrixBuilder extends MatrixBuilder {

    @Nonnull
    private final IntArrayList rowPointers;
    @Nonnull
    private final IntArrayList columnIndices;
    @Nonnull
    private final DoubleArrayList values;

    private int maxNumColumns;

    public CSRMatrixBuilder(@Nonnegative int initSize) {
        super();
        this.rowPointers = new IntArrayList(initSize + 1);
        rowPointers.add(0);
        this.columnIndices = new IntArrayList(initSize);
        this.values = new DoubleArrayList(initSize);
        this.maxNumColumns = 0;
    }

    @Override
    public CSRMatrixBuilder nextRow() {
        int ptr = values.size();
        rowPointers.add(ptr);
        return this;
    }

    @Override
    public CSRMatrixBuilder nextColumn(@Nonnegative int col, double value) {
        if (value == 0.d) {
            return this;
        }

        columnIndices.add(col);
        values.add(value);
        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        return this;
    }

    @Override
    public CSRMatrix buildMatrix() {
        CSRMatrix matrix = new CSRMatrix(rowPointers.toArray(true), columnIndices.toArray(true),
            values.toArray(true), maxNumColumns);
        return matrix;
    }

}
