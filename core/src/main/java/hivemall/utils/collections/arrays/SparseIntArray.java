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

import hivemall.utils.function.Consumer;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class SparseIntArray implements IntArray {
    private static final long serialVersionUID = -2814248784231540118L;

    private int[] mKeys;
    private int[] mValues;
    private int mSize;

    public SparseIntArray() {
        this(10);
    }

    public SparseIntArray(@Nonnegative int initialCapacity) {
        this.mKeys = new int[initialCapacity];
        this.mValues = new int[initialCapacity];
        this.mSize = 0;
    }

    public SparseIntArray(@Nonnull final int[] values) {
        this.mKeys = MathUtils.permutation(values.length);
        this.mValues = values;
        this.mSize = values.length;
    }

    private SparseIntArray(@Nonnull int[] mKeys, @Nonnull int[] mValues, @Nonnegative int mSize) {
        this.mKeys = mKeys;
        this.mValues = mValues;
        this.mSize = mSize;
    }

    public IntArray deepCopy() {
        int[] newKeys = new int[mSize];
        int[] newValues = new int[mSize];
        System.arraycopy(mKeys, 0, newKeys, 0, mSize);
        System.arraycopy(mValues, 0, newValues, 0, mSize);
        return new SparseIntArray(newKeys, newValues, mSize);
    }

    @Override
    public int get(int key) {
        return get(key, 0);
    }

    @Override
    public int get(int key, int valueIfKeyNotFound) {
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
    public void put(int key, int value) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i >= 0) {
            this.mValues[i] = value;
        } else {
            i = ~i;
            this.mKeys = ArrayUtils.insert(mKeys, mSize, i, key);
            this.mValues = ArrayUtils.insert(mValues, mSize, i, value);
            this.mSize++;
        }
    }

    @Override
    public void increment(int key, int value) {
        int i = Arrays.binarySearch(mKeys, 0, mSize, key);
        if (i >= 0) {
            this.mValues[i] += value;
        } else {
            i = ~i;
            this.mKeys = ArrayUtils.insert(mKeys, mSize, i, key);
            this.mValues = ArrayUtils.insert(mValues, mSize, i, value);
            this.mSize++;
        }
    }

    @Override
    public int size() {
        return mSize;
    }

    public int firstKey() {
        if (mKeys.length == 0) {
            return -1;
        }
        return mKeys[0];
    }

    public int lastKey() {
        if (mKeys.length == 0) {
            return -1;
        }
        return mKeys[mKeys.length - 1];
    }

    @Override
    public int keyAt(int index) {
        return mKeys[index];
    }

    public int valueAt(int index) {
        return mValues[index];
    }

    public void setValueAt(int index, int value) {
        this.mValues[index] = value;
    }

    public int indexOfKey(int key) {
        return Arrays.binarySearch(mKeys, 0, mSize, key);
    }

    public int indexOfValue(int value) {
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
        this.mSize = 0;
        if (zeroFill) {
            Arrays.fill(mKeys, 0);
            Arrays.fill(mValues, 0);
        }
    }

    public void append(int key, int value) {
        if (mSize != 0 && key <= mKeys[mSize - 1]) {
            put(key, value);
            return;
        }
        this.mKeys = ArrayUtils.append(mKeys, mSize, key);
        this.mValues = ArrayUtils.append(mValues, mSize, value);
        this.mSize++;
    }

    public void append(@Nonnegative final int dstPos, @Nonnull final int[] values) {
        if (mSize == 0) {
            this.mKeys = MathUtils.permutation(dstPos, values.length);
            this.mValues = values.clone();
            this.mSize = values.length;
            return;
        }

        final int lastKey = mKeys[mSize - 1];
        for (int i = 0; i < values.length; i++) {
            final int key = dstPos + i;
            if (key <= lastKey) {
                put(key, values[i]);
            } else {// append
                int length = values.length - i;
                this.mKeys = ArrayUtils.concat(mKeys, MathUtils.permutation(key, length));
                this.mValues = ArrayUtils.concat(mValues, values, i, length);
                this.mSize += length;
                break;
            }
        }
    }

    public void append(@Nonnegative final int dstPos, @Nonnull final int[] values, final int offset,
            final int length) {
        if (mSize == 0) {
            this.mKeys = MathUtils.permutation(dstPos, length);
            this.mValues = Arrays.copyOfRange(values, offset, length);
            this.mSize = length;
            return;
        }

        final int lastKey = mKeys[mSize - 1];
        for (int i = 0; i < length; i++) {
            final int valuePos = offset + i;
            final int key = dstPos + i;
            if (key <= lastKey) {
                put(key, values[valuePos]);
            } else {// append
                int size = length - i;
                this.mKeys = ArrayUtils.concat(mKeys, MathUtils.permutation(key, size));
                this.mValues = ArrayUtils.concat(mValues, values, valuePos, size);
                this.mSize += size;
                break;
            }
        }
    }

    public void consume(@Nonnegative final int start, @Nonnegative final int end,
            @Nonnull final Consumer consumer) {
        int startPos = indexOfKey(start);
        if (startPos < 0) {
            startPos = ~startPos;
        }
        int endPos = indexOfKey(end);
        if (endPos < 0) {
            endPos = ~endPos;
        }
        // mKeys, mValues may be replaced by in-place update
        final int[] keys = mKeys.clone();
        final int[] values = mValues.clone();
        for (int i = startPos; i < endPos; i++) {
            int k = keys[i];
            int v = values[i];
            consumer.accept(k, v);
        }
    }

    @Nonnull
    public int[] toArray() {
        return toArray(true);
    }

    @Override
    public int[] toArray(boolean copy) {
        if (mSize == 0) {
            return new int[0];
        }

        int last = mKeys[mSize - 1];
        final int[] array = new int[last + 1];
        for (int i = 0; i < mSize; i++) {
            int k = mKeys[i];
            int v = mValues[i];
            Preconditions.checkArgument(k >= 0, "Negative key is not allowed for toArray(): " + k);
            array[k] = v;
        }
        return array;
    }

    @Override
    public String toString() {
        if (mSize == 0) {
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
            int value = valueAt(i);
            buffer.append(value);
        }
        buffer.append('}');
        return buffer.toString();
    }


}
