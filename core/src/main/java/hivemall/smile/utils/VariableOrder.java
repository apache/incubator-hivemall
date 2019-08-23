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
package hivemall.smile.utils;

import hivemall.utils.collections.arrays.SparseIntArray;
import hivemall.utils.function.Consumer;

import javax.annotation.Nonnull;

public final class VariableOrder {

    @Nonnull
    private final SparseIntArray[] cols; // col => row

    public VariableOrder(@Nonnull SparseIntArray[] cols) {
        this.cols = cols;
    }

    public void eachRow(@Nonnull final Consumer consumer) {
        for (int j = 0; j < cols.length; j++) {
            final SparseIntArray row = cols[j];
            if (row == null) {
                continue;
            }
            consumer.accept(j, row);
        }
    }

    public void eachNonNullInColumn(final int col, final int startRow, final int endRow,
            @Nonnull final Consumer consumer) {
        final SparseIntArray row = cols[col];
        if (row == null) {
            return;
        }
        row.forEach(startRow, endRow, consumer);
    }

}
