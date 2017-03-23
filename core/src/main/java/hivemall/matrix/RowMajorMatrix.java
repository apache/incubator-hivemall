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

import hivemall.vector.Vector;
import hivemall.vector.VectorProcedure;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public abstract class RowMajorMatrix extends AbstractMatrix {

    public RowMajorMatrix() {
        super();
    }

    @Override
    public boolean isRowMajorMatrix() {
        return true;
    }

    @Override
    public boolean isColumnMajorMatrix() {
        return false;
    }

    @Override
    public void getRow(@Nonnegative final int index, @Nonnull final Vector row) {
        row.clear();
        eachInRow(index, new VectorProcedure() {
            @Override
            public void apply(final int i, final double value) {
                row.set(i, value);
            }
        }, false);
    }

    @Override
    public void eachInColumn(int col, VectorProcedure procedure, boolean nullOutput) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void eachInNonZeroColumn(int col, VectorProcedure procedure) {
        throw new UnsupportedOperationException();
    }

    @Override
    public RowMajorMatrix toRowMajorMatrix() {
        return this;
    }

}