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
package hivemall.math.matrix;

import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.matrix.ints.IntMatrix;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.mutable.MutableInt;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class MatrixUtils {

    private MatrixUtils() {}

    @Nonnull
    public static Matrix shuffle(@Nonnull final Matrix m, @Nonnull final int[] indices) {
        Preconditions.checkArgument(m.numRows() <= indices.length, "m.numRow() `" + m.numRows()
                + "` MUST be equals to or less than |swapIndicies| `" + indices.length + "`");

        final MatrixBuilder builder = m.builder();
        final VectorProcedure proc = new VectorProcedure() {
            public void apply(int col, double value) {
                builder.nextColumn(col, value);
            }
        };
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            m.eachNonNullInRow(idx, proc);
            builder.nextRow();
        }
        return builder.buildMatrix();
    }

    /**
     * Returns the index of maximum value of an array.
     * 
     * @return -1 if there are no columns
     */
    public static int whichMax(@Nonnull final IntMatrix matrix, @Nonnegative final int row) {
        final MutableInt m = new MutableInt(Integer.MIN_VALUE);
        final MutableInt which = new MutableInt(-1);
        matrix.eachInRow(row, new VectorProcedure() {
            @Override
            public void apply(int i, int value) {
                if (value > m.getValue()) {
                    m.setValue(value);
                    which.setValue(i);
                }
            }
        }, false);
        return which.getValue();
    }

}
