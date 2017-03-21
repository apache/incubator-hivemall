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
package hivemall.utils.collections.sets;

import hivemall.utils.lang.ArrayUtils;

import java.util.Arrays;

import javax.annotation.Nonnull;

public final class IntArraySet implements IntSet {

    @Nonnull
    private int[] mKeys;
    private int mSize;

    public IntArraySet() {
        this(0);
    }

    public IntArraySet(int initSize) {
        this.mKeys = new int[initSize];
        this.mSize = 0;
    }

    @Override
    public boolean add(final int k) {
        final int i = Arrays.binarySearch(mKeys, 0, mSize, k);
        if (i >= 0) {
            return false;
        }
        mKeys = ArrayUtils.insert(mKeys, mSize, ~i, k);
        mSize++;
        return true;
    }

    @Override
    public boolean remove(final int k) {
        final int i = Arrays.binarySearch(mKeys, 0, mSize, k);
        if (i < 0) {
            return false;
        }
        System.arraycopy(mKeys, i + 1, mKeys, i, mSize - (i + 1));
        mSize--;
        return true;
    }

    @Override
    public boolean contains(final int k) {
        return Arrays.binarySearch(mKeys, 0, mSize, k) >= 0;
    }

    @Override
    public int size() {
        return mSize;
    }

    @Override
    public void clear() {
        this.mSize = 0;
    }

    @Override
    public int[] toArray(final boolean copy) {
        if (copy == false && mKeys.length == mSize) {
            return mKeys;
        }

        return Arrays.copyOf(mKeys, mSize);
    }

}
