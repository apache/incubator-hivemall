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
//
//   Copyright (C) 2010 catchpole.net
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
package hivemall.utils.collections.maps;

import hivemall.utils.hashing.HashUtils;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;

import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

/**
 * A space efficient open-addressing HashMap implementation with integer keys and long values.
 * 
 * Unlike {@link Int2LongOpenHashTable}, it maintains single arrays for keys and object references.
 * 
 * It uses single open hashing arrays sized to binary powers (256, 512 etc) rather than those
 * divisible by prime numbers. This allows the hash offset calculation to be a simple binary masking
 * operation.
 * 
 * The index into the arrays is determined by masking a portion of the key and shifting it to
 * provide a series of small buckets within the array. To insert an entry the a sweep is searched
 * until an empty key space is found. A sweep is 4 times the length of a bucket, to reduce the need
 * to rehash. If no key space is found within a sweep, the table size is doubled.
 * 
 * While performance is high, the slowest situation is where lookup occurs for entries that do not
 * exist, as an entire sweep area must be searched. However, this HashMap is more space efficient
 * than other open-addressing HashMap implementations as in fastutil.
 */
@NotThreadSafe
public final class Int2LongOpenHashMap {

    // special treatment for key=0
    private boolean hasKey0 = false;
    private long value0 = 0L;

    private int[] keys;
    private long[] values;

    // total number of entries in this table
    private int size;
    // number of bits for the value table (eg. 8 bits = 256 entries)
    private int bits;
    // the number of bits in each sweep zone.
    private int sweepbits;
    // the size of a sweep (2 to the power of sweepbits)
    private int sweep;
    // the sweepmask used to create sweep zone offsets
    private int sweepmask;

    public Int2LongOpenHashMap(int size) {
        resize(MathUtils.bitsRequired(size < 256 ? 256 : size));
    }

    public long put(final int key, final long value) {
        if (key == 0) {
            if (!hasKey0) {
                this.hasKey0 = true;
                size++;
            }
            long old = value0;
            this.value0 = value;
            return old;
        }

        for (;;) {
            int off = getBucketOffset(key);
            final int end = off + sweep;
            for (; off < end; off++) {
                final int searchKey = keys[off];
                if (searchKey == 0) { // insert
                    keys[off] = key;
                    size++;
                    long previous = values[off];
                    values[off] = value;
                    return previous;
                } else if (searchKey == key) {// replace
                    long previous = values[off];
                    values[off] = value;
                    return previous;
                }
            }
            resize(this.bits + 1);
        }
    }

    public long putIfAbsent(final int key, final long value) {
        if (key == 0) {
            if (hasKey0) {
                return value0;
            }
            this.hasKey0 = true;
            long old = value0;
            this.value0 = value;
            size++;
            return old;
        }

        for (;;) {
            int off = getBucketOffset(key);
            final int end = off + sweep;
            for (; off < end; off++) {
                final int searchKey = keys[off];
                if (searchKey == 0) { // insert
                    keys[off] = key;
                    size++;
                    long previous = values[off];
                    values[off] = value;
                    return previous;
                } else if (searchKey == key) {// replace
                    return values[off];
                }
            }
            resize(this.bits + 1);
        }
    }

    public long get(final int key) {
        return get(key, 0L);
    }

    public long get(final int key, final long defaultValue) {
        if (key == 0) {
            return hasKey0 ? value0 : defaultValue;
        }

        int off = getBucketOffset(key);
        final int end = sweep + off;
        for (; off < end; off++) {
            if (keys[off] == key) {
                return values[off];
            }
        }
        return defaultValue;
    }

    public long remove(final int key, final long defaultValue) {
        if (key == 0) {
            if (hasKey0) {
                this.hasKey0 = false;
                long old = value0;
                this.value0 = 0L;
                size--;
                return old;
            } else {
                return defaultValue;
            }
        }

        int off = getBucketOffset(key);
        final int end = sweep + off;
        for (; off < end; off++) {
            if (keys[off] == key) {
                keys[off] = 0;
                long previous = values[off];
                values[off] = 0L;
                size--;
                return previous;
            }
        }
        return defaultValue;
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public boolean containsKey(final int key) {
        if (key == 0) {
            return hasKey0;
        }

        int off = getBucketOffset(key);
        final int end = sweep + off;
        for (; off < end; off++) {
            if (keys[off] == key) {
                return true;
            }
        }
        return false;
    }

    public void clear() {
        this.hasKey0 = false;
        this.value0 = 0L;
        Arrays.fill(keys, 0);
        Arrays.fill(values, 0L);
        this.size = 0;
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + ' ' + size;
    }

    private void resize(final int bits) {
        this.bits = bits;
        this.sweepbits = bits / 4;
        this.sweep = MathUtils.powerOf(2, sweepbits) * 4;
        this.sweepmask = MathUtils.bitMask(bits - sweepbits) << sweepbits;

        // remember old values so we can recreate the entries
        final int[] existingKeys = this.keys;
        final long[] existingValues = this.values;

        // create the arrays
        this.values = new long[MathUtils.powerOf(2, bits) + sweep];
        this.keys = new int[values.length];
        this.size = hasKey0 ? 1 : 0;

        // re-add the previous entries if resizing
        if (existingKeys != null) {
            for (int i = 0; i < existingKeys.length; i++) {
                final int k = existingKeys[i];
                if (k != 0) {
                    put(k, existingValues[i]);
                }
            }
        }
    }

    private int getBucketOffset(final int key) {
        return (HashUtils.fnv1a(key) << sweepbits) & sweepmask;
    }

    @Nonnull
    public MapIterator entries() {
        return new MapIterator();
    }

    public final class MapIterator {

        int nextEntry;
        int lastEntry = -2;

        MapIterator() {
            this.nextEntry = nextEntry(-1);
        }

        /** find the index of next full entry */
        int nextEntry(int index) {
            if (index == -1) {
                if (hasKey0) {
                    return -1;
                } else {
                    index = 0;
                }
            }
            while (index < keys.length && keys[index] == 0) {
                index++;
            }
            return index;
        }

        public boolean hasNext() {
            return nextEntry < keys.length;
        }

        public boolean next() {
            free(lastEntry);
            if (!hasNext()) {
                return false;
            }
            int curEntry = nextEntry;
            this.lastEntry = curEntry;
            this.nextEntry = nextEntry(curEntry + 1);
            return true;
        }

        public int getKey() {
            if (lastEntry >= 0 && lastEntry < keys.length) {
                return keys[lastEntry];
            } else if (lastEntry == -1) {
                return 0;
            } else {
                throw new IllegalStateException(
                    "next() should be called before getKey(). lastEntry=" + lastEntry
                            + ", keys.length=" + keys.length);
            }
        }

        public long getValue() {
            if (lastEntry >= 0 && lastEntry < keys.length) {
                return values[lastEntry];
            } else if (lastEntry == -1) {
                return value0;
            } else {
                throw new IllegalStateException(
                    "next() should be called before getKey(). lastEntry=" + lastEntry
                            + ", keys.length=" + keys.length);
            }
        }

        private void free(int index) {
            if (index >= 0) {
                if (index >= keys.length) {
                    throw new IllegalStateException("index=" + index + ", keys.length="
                            + keys.length);
                }
                keys[index] = 0;
                values[index] = 0L;
            } else if (index == -1) {
                hasKey0 = false;
                value0 = 0L;
            }
            // index may be -2
        }

    }
}
