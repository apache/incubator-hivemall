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

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public abstract class DMatrixBuilder {

    public DMatrixBuilder() {}

    protected static final void checkColIndex(final int col) {
        if (col < 0) {
            throw new IllegalArgumentException("Found negative column index: " + col);
        }
    }

    public void nextRow(@Nonnull final float[] row) {
        for (int col = 0; col < row.length; col++) {
            nextColumn(col, row[col]);
        }
        nextRow();
    }

    public void nextRow(@Nonnull final String[] row) {
        for (String col : row) {
            if (col == null) {
                continue;
            }
            nextColumn(col);
        }
        nextRow();
    }

    public void nextRow(@Nonnull final String[] row, final int start, final int endEx) {
        for (int i = start, last = Math.min(endEx, row.length); i < last; i++) {
            String col = row[i];
            if (col == null) {
                continue;
            }
            nextColumn(col);
        }
        nextRow();
    }

    @Nonnull
    public abstract DMatrixBuilder nextRow();

    @Nonnull
    public abstract DMatrixBuilder nextColumn(@Nonnegative int col, float value);

    /**
     * @throws IllegalArgumentException
     * @throws NumberFormatException
     */
    @Nonnull
    public DMatrixBuilder nextColumn(@Nonnull final String col) {
        final int pos = col.indexOf(':');
        if (pos == 0) {
            throw new IllegalArgumentException("Invalid feature value representation: " + col);
        }

        final String feature;
        final float value;
        if (pos > 0) {
            feature = col.substring(0, pos);
            String s2 = col.substring(pos + 1);
            value = Float.parseFloat(s2);
        } else {
            feature = col;
            value = 1.f;
        }

        if (feature.indexOf(':') != -1) {
            throw new IllegalArgumentException("Invalid feature format `<index>:<value>`: " + col);
        }

        int colIndex = Integer.parseInt(feature);
        if (colIndex < 0) {
            throw new IllegalArgumentException(
                "Col index MUST be greater than or equals to 0: " + colIndex);
        }

        return nextColumn(colIndex, value);
    }

    @Nonnull
    public abstract DMatrix buildMatrix(@Nonnull float[] labels) throws XGBoostError;

}
