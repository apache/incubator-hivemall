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
package hivemall.math.vector;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class DenseVector extends AbstractVector {

    @Nonnull
    private final double[] values;
    private final int size;

    public DenseVector(@Nonnegative int size) {
        super();
        this.values = new double[size];
        this.size = size;
    }

    public DenseVector(@Nonnull double[] values) {
        super();
        this.values = values;
        this.size = values.length;
    }

    @Override
    public double get(@Nonnegative final int index, final double defaultValue) {
        checkIndex(index);
        if (index >= size) {
            return defaultValue;
        }

        return values[index];
    }

    @Override
    public void set(@Nonnegative final int index, final double value) {
        checkIndex(index, size);

        values[index] = value;
    }

    @Override
    public void each(@Nonnull final VectorProcedure procedure) {
        for (int i = 0; i < values.length; i++) {
            procedure.apply(i, values[i]);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void clear() {
        Arrays.fill(values, 0.d);
    }

}
