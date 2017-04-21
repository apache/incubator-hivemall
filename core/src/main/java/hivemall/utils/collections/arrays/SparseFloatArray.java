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

import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.Preconditions;

import java.util.Arrays;

import javax.annotation.Nonnull;

public final class SparseFloatArray implements FloatArray {
    private static final long serialVersionUID = -2814248784231540118L;

    private int[] mKeys;
    private float[] mValues;
    private int mSize;

    public SparseFloatArray() {
        this(10);
    }

    public SparseFloatArray(int initialCapacity) {
        mKeys = new int[initialCapacity];
        mValues = new float[initialCapacity];
        mSize = 0;
    }

    private SparseFloatArray(@Nonnull int[] mKeys, @Nonnull float[] mValues, int mSize) {
        this.mKeys = mKeys;
        this.mValues = mValues;
        this.mSize = mSize;
    }

    public SparseFloatArray deepCopy() {
        int[] newKeys = new int[mSize];
        float[] newValues = new float[mSize];
        System.arraycopy(mKeys, 0, newKeys, 0, mSize);
        System.arraycopy(mValues, 0, newValues, 0, mSize);
        return new SparseFloatArray(newKeys, newValues, mSize);
    }

    @Override
    public float get(int key) {
        return get(key, 0.f);
    }

    @Override
    public float get(int key, float valueIfKeyNotFound) {
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
    public void put(int key, float value) {
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

    public void increment(int key, float value) {
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

    public float valueAt(int index) {
        return mValues[index];
    }

    public void setValueAt(int index, float value) {
        mValues[index] = value;
    }

    public int indexOfKey(int key) {
        return Arrays.binarySearch(mKeys, 0, mSize, key);
    }

    public int indexOfValue(float value) {
        for (int i = 0; i < mSize; i++) {
            if (mValues[i] == value) {
                return i;
            }
        }
        return -1;
    }

    public void clear() {
        clear(true);
    }

    public void clear(boolean zeroFill) {
        mSize = 0;
        if (zeroFill) {
            Arrays.fill(mKeys, 0);
            Arrays.fill(mValues, 0.f);
        }
    }

    public void append(int key, float value) {
        if (mSize != 0 && key <= mKeys[mSize - 1]) {
            put(key, value);
            return;
        }
        mKeys = ArrayUtils.append(mKeys, mSize, key);
        mValues = ArrayUtils.append(mValues, mSize, value);
        mSize++;
    }

    @Nonnull
    public float[] toArray() {
        return toArray(true);
    }

    @Override
    public float[] toArray(boolean copy) {
        if (mSize == 0) {
            return new float[0];
        }

        int last = mKeys[mSize - 1];
        final float[] array = new float[last + 1];
        for (int i = 0; i < mSize; i++) {
            int k = mKeys[i];
            float v = mValues[i];
            Preconditions.checkArgument(k >= 0, "Negative key is not allowed for toArray(): " + k);
            array[k] = v;
        }
        return array;
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
            float value = valueAt(i);
            buffer.append(value);
        }
        buffer.append('}');
        return buffer.toString();
    }


}
