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
package hivemall.utils.collections.arrays;

import java.util.Arrays;

import javax.annotation.Nonnull;

/**
 * A fixed double array that has keys greater than or equals to 0.
 */
public final class DenseDoubleArray implements DoubleArray {
    private static final long serialVersionUID = 4282904528662802088L;

    @Nonnull
    private final double[] array;
    private final int size;

    public DenseDoubleArray(@Nonnull int size) {
        this.array = new double[size];
        this.size = size;
    }

    public DenseDoubleArray(@Nonnull double[] array) {
        this.array = array;
        this.size = array.length;
    }

    @Override
    public double get(int index) {
        return array[index];
    }

    @Override
    public double get(int index, double valueIfKeyNotFound) {
        if (index >= size) {
            return valueIfKeyNotFound;
        }
        return array[index];
    }

    @Override
    public void put(int index, double value) {
        array[index] = value;
    }

    @Override
    public int size() {
        return array.length;
    }

    @Override
    public int keyAt(int index) {
        return index;
    }

    @Override
    public double[] toArray() {
        return toArray(true);
    }

    @Override
    public double[] toArray(boolean copy) {
        if (copy) {
            return Arrays.copyOf(array, size);
        } else {
            return array;
        }
    }

    @Override
    public void clear() {
        Arrays.fill(array, 0.d);
    }

}
