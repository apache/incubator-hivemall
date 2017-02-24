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

import hivemall.utils.lang.Preconditions;

import javax.annotation.Nonnull;

public final class MatrixUtils {

    private MatrixUtils() {}

    @Nonnull
    public static Matrix shuffle(@Nonnull final Matrix m, @Nonnull final int[] swapIndices) {
        Preconditions.checkArgument(m.numRows() <= swapIndices.length, "m.numRow() `" + m.numRows()
                + "` MUST be equals to or less than |swapIndicies| `" + swapIndices.length + "`");

        if (m.shufflable()) {
            for (int i = 0; i < swapIndices.length; i++) {
                int j = swapIndices[i];
                m.swap(i, j);
            }
            return m;
        }

        final MatrixBuilder builder = m.builder();
        for (int i = 0; i < swapIndices.length; i++) {
            int j = swapIndices[i];
            m.eachNonZeroInRow(j, new VectorProcedure() {
                public void apply(int col, double value) {
                    builder.nextColumn(col, value);
                }
            });
            builder.nextRow();
        }
        return builder.buildMatrix();
    }

}
