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
package hivemall.utils.collections.lists;

import hivemall.utils.lang.ArrayUtils;

import java.io.Serializable;
import java.util.NoSuchElementException;
import java.util.Objects;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;

public final class IntArrayList implements Serializable {
    private static final long serialVersionUID = -2147675120406747488L;
    public static final int DEFAULT_CAPACITY = 12;

    /** array entity */
    @Nonnull
    private int[] data;
    private int used;

    public IntArrayList() {
        this(DEFAULT_CAPACITY);
    }

    public IntArrayList(int size) {
        this.data = new int[size];
        this.used = 0;
    }

    public IntArrayList(@CheckForNull int[] initValues) {
        this.data = Objects.requireNonNull(initValues);
        this.used = initValues.length;
    }

    @Nonnull
    public IntArrayList add(final int value) {
        if (used >= data.length) {
            expand(used + 1);
        }
        data[used++] = value;
        return this;
    }

    @Nonnull
    public IntArrayList add(@Nonnull final int[] values) {
        final int needs = used + values.length;
        if (needs >= data.length) {
            expand(needs);
        }
        System.arraycopy(values, 0, data, used, values.length);
        this.used = needs;
        return this;
    }

    /**
     * dynamic expansion.
     */
    private void expand(final int minimumCapacity) {
        while (data.length < minimumCapacity) {
            int oldLen = data.length;
            int newLen = (int) Math.max(minimumCapacity, Math.min(oldLen * 2L, Integer.MAX_VALUE));
            int[] newArray = new int[newLen];
            System.arraycopy(data, 0, newArray, 0, oldLen);
            this.data = newArray;
        }
    }

    public int remove() {
        if (used == 0) {
            throw new NoSuchElementException("No elements to remove");
        }
        return data[--used];
    }

    public int remove(final int index) {
        if (index >= used) {
            throw new IndexOutOfBoundsException();
        }

        final int ret;
        if (index == used) {
            ret = data[index];
            --used;
        } else { // index < used
            ret = data[index];
            System.arraycopy(data, index + 1, data, index, used - index - 1);
            --used;
        }
        return ret;
    }

    public void set(final int index, final int value) {
        if (index > used) {
            throw new IllegalArgumentException(
                "Index " + index + " MUST be less than size() " + used);
        } else if (index == used) {
            ++used;
        }
        data[index] = value;
    }

    public int get(final int index) {
        if (index >= used) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds " + used);
        }
        return data[index];
    }

    public int fastGet(final int index) {
        return data[index];
    }

    /**
     * @return -1 if not found.
     */
    public int indexOf(final int key) {
        return ArrayUtils.indexOf(data, key, 0, used);
    }

    public boolean contains(final int key) {
        return ArrayUtils.indexOf(data, key, 0, used) != -1;
    }

    public int size() {
        return used;
    }

    public boolean isEmpty() {
        return used == 0;
    }

    public void clear() {
        used = 0;
    }

    @Nonnull
    public int[] toArray() {
        return toArray(false);
    }

    @Nonnull
    public int[] toArray(boolean close) {
        final int[] newArray = new int[used];
        System.arraycopy(data, 0, newArray, 0, used);
        if (close) {
            this.data = null;
        }
        return newArray;
    }

    public int[] array() {
        return data;
    }

    @Override
    public String toString() {
        final StringBuilder buf = new StringBuilder();
        buf.append('[');
        for (int i = 0; i < used; i++) {
            if (i != 0) {
                buf.append(", ");
            }
            buf.append(data[i]);
        }
        buf.append(']');
        return buf.toString();
    }

}
