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
package hivemall.vector;

import hivemall.utils.collections.arrays.SparseDoubleArray;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class SparseVector extends AbstractVector {

    @Nonnull
    private final SparseDoubleArray values;

    public SparseVector() {
        super();
        this.values = new SparseDoubleArray();
    }

    public SparseVector(@Nonnull SparseDoubleArray values) {
        super();
        this.values = values;
    }

    @Override
    public double get(@Nonnegative final int index, final double defaultValue) {
        return values.get(index, defaultValue);
    }

    @Override
    public void set(@Nonnegative final int index, final double value) {
        values.put(index, value);
    }

    @Override
    public void each(@Nonnull final VectorProcedure procedure) {
        values.each(procedure);
    }

    @Override
    public int size() {
        return values.size();
    }

    @Override
    public void clear() {
        values.clear();
    }

}
