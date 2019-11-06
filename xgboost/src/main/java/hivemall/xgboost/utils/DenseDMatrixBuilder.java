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

import hivemall.utils.collections.arrays.SparseFloatArray;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class DenseDMatrixBuilder extends DMatrixBuilder {

    @Nonnull
    private final List<float[]> rows;
    private int maxNumColumns;

    @Nonnull
    private final SparseFloatArray rowProbe;

    public DenseDMatrixBuilder(@Nonnegative int initSize) {
        super();
        this.rows = new ArrayList<float[]>(initSize);
        this.maxNumColumns = 0;
        this.rowProbe = new SparseFloatArray(32);
    }

    @Override
    public DenseDMatrixBuilder nextColumn(@Nonnegative final int col, final float value) {
        checkColIndex(col);

        this.maxNumColumns = Math.max(col + 1, maxNumColumns);
        if (value == 0.d) {
            return this;
        }
        rowProbe.put(col, value);
        return this;
    }

    @Override
    public DenseDMatrixBuilder nextRow() {
        float[] row = rowProbe.toArray();
        rowProbe.clear();
        rows.add(row);
        return this;
    }

    @Override
    public DMatrix buildMatrix(@Nonnull float[] labels) throws XGBoostError {
        final int numRows = rows.size();
        if (labels.length != numRows) {
            throw new XGBoostError(
                String.format("labels.length does not match to nrows. labels.length=%d, nrows=%d",
                    labels.length, numRows));
        }

        final float[] data = new float[numRows * maxNumColumns];
        Arrays.fill(data, Float.NaN);
        for (int i = 0; i < numRows; i++) {
            final float[] row = rows.get(i);
            final int rowPtr = i * maxNumColumns;
            for (int j = 0; j < row.length; j++) {
                int ij = rowPtr + j;
                data[ij] = row[j];
            }
        }

        DMatrix matrix = new DMatrix(data, numRows, maxNumColumns, Float.NaN);
        matrix.setLabel(labels);
        return matrix;
    }

}
