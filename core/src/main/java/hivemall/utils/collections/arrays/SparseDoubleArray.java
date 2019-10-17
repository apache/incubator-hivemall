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

import matrix4j.vector.VectorProcedure;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.Preconditions;

import java.util.Arrays;

import javax.annotation.Nonnull;

public final class SparseDoubleArray implements DoubleArray {
    private static final long serialVersionUID = -2814248784231540118L;

    @Nonnull
    private int[] mKeys;
    @Nonnull
    private double[] mValues;
    private int mSize;

    public SparseDoubleArray() {
        this(10);
    }

    public SparseDoubleArray(int initialCapacity) {
        mKeys = new int[initialCapacity];
        mValues = new double[initialCapacity];
        mSize = 0;
    }

    private SparseDoubleArray(@Nonnull int[] mKeys, @Nonnull double[] mValues, int mSize) {
        this.mKeys = mKeys;
        this.mValues = mValues;
        this.mSize = mSize;
    }

    @Nonnull
    public SparseDoubleArray deepCopy() {
        int[] newKeys = new int[mSize];
        double[] newValues = new double[mSize];
        System.arraycopy(mKeys, 0, newKeys, 0, mSize);
        System.arraycopy(mValues, 0, newValues, 0, mSize);
        return new SparseDoubleArray(newKeys, newValues, mSize);
    }

    @Override
    public double get(int key) {
        return get(key, 0);
    }

    @Override
    public double get(int key, double valueIfKeyNotFound) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i < 0) {
            return valueIfKeyNotFound;
        } else {
            return mValues[i];
        }
    }

    public void delete(int key) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i >= 0) {
            removeAt(i);
        }
    }

    public void removeAt(int index) {
        System.arraycopy(mKeys, index + 1, mKeys, index, mSize - (index + 1));
        System.arraycopy(mValues, index + 1, mValues, index, mSize - (index + 1));
        mSize--;
    }

    @Override
    public void put(int key, double value) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i >= 0) {
            mValues[i] = value;
        } else {
            i = ~i;
            mKeys = ArrayUtils.insert(mKeys, mSize, i, key);
            mValues = ArrayUtils.insert(mValues, mSize, i, value);
            mSize++;
        }
    }

    public void increment(int key, double value) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i >= 0) {
            mValues[i] += value;
        } else {
            i = ~i;
            mKeys = ArrayUtils.insert(mKeys, mSize, i, key);
            mValues = ArrayUtils.insert(mValues, mSize, i, value);
            mSize++;
        }
    }

    @Override
    public int size() {
        return mSize;
    }

    @Override
    public int keyAt(int index) {
        return mKeys[index];
    }

    public double valueAt(int index) {
        return mValues[index];
    }

    public void setValueAt(int index, double value) {
        mValues[index] = value;
    }

    public int indexOfKey(int key) {
        return Arrays.binarySearch(mKeys, 0, mSize, key);
    }

    public int indexOfValue(double value) {
        for (int i = 0; i < mSize; i++) {
            if (mValues[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public void clear() {
        clear(true);
    }

    public void clear(boolean zeroFill) {
        mSize = 0;
        if (zeroFill) {
            Arrays.fill(mKeys, 0);
            Arrays.fill(mValues, 0.d);
        }
    }

    public void append(int key, double value) {
        if (mSize != 0 && key <= mKeys[mSize - 1]) {
            put(key, value);
            return;
        }
        mKeys = ArrayUtils.append(mKeys, mSize, key);
        mValues = ArrayUtils.append(mValues, mSize, value);
        mSize++;
    }

    @Override
    public double[] toArray() {
        return toArray(true);
    }

    @Override
    public double[] toArray(boolean copy) {
        if (mSize == 0) {
            return new double[0];
        }

        int last = mKeys[mSize - 1];
        final double[] array = new double[last + 1];
        for (int i = 0; i < mSize; i++) {
            int k = mKeys[i];
            double v = mValues[i];
            Preconditions.checkArgument(k >= 0, "Negative key is not allowed for toArray(): " + k);
            array[k] = v;
        }
        return array;
    }

    public void each(@Nonnull final VectorProcedure procedure) {
        for (int i = 0; i < mSize; i++) {
            int k = mKeys[i];
            double v = mValues[i];
            procedure.apply(k, v);
        }
    }

    @Override
    public String toString() {
        if (size() <= 0) {
            return "{}";
        }

        StringBuilder buffer = new StringBuilder(mSize * 28);
        buffer.append('{');
        for (int i = 0; i < mSize; i++) {
            if (i > 0) {
                buffer.append(", ");
            }
            int key = keyAt(i);
            buffer.append(key);
            buffer.append('=');
            double value = valueAt(i);
            buffer.append(value);
        }
        buffer.append('}');
        return buffer.toString();
    }


}
