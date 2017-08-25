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

import hivemall.utils.collections.IMapIterator;
import hivemall.utils.lang.Copyable;
import hivemall.utils.math.MathUtils;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * A space efficient open-addressing HashMap implementation.
 * 
 * Unlike {@link OpenHashTable}, it maintains single arrays for keys and object references.
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
 *
 * Note that this HashMap does not allow nulls to be used as keys.
 */
public final class OpenHashMap<K, V> implements Map<K, V>, Externalizable {
    private K[] keys;
    private V[] values;

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

    public OpenHashMap() {}// for Externalizable

    public OpenHashMap(int size) {
        resize(MathUtils.bitsRequired(size < 256 ? 256 : size));
    }

    @Nullable
    public V put(@CheckForNull final K key, @Nullable final V value) {
        if (key == null) {
            throw new NullPointerException(this.getClass().getName() + " key");
        }

        for (;;) {
            int off = getBucketOffset(key);
            final int end = off + sweep;
            for (; off < end; off++) {
                final K searchKey = keys[off];
                if (searchKey == null) {
                    // insert
                    keys[off] = key;
                    size++;
                    V previous = values[off];
                    values[off] = value;
                    return previous;
                } else if (compare(searchKey, key)) {
                    // replace
                    V previous = values[off];
                    values[off] = value;
                    return previous;
                }
            }
            resize(this.bits + 1);
        }
    }

    @Nullable
    public V putIfAbsent(@CheckForNull final K key, @Nullable final V value) {
        if (key == null) {
            throw new NullPointerException(this.getClass().getName() + " key");
        }

        for (;;) {
            int off = getBucketOffset(key);
            final int end = off + sweep;
            for (; off < end; off++) {
                final K searchKey = keys[off];
                if (searchKey == null) {
                    // insert
                    keys[off] = key;
                    size++;
                    V previous = values[off];
                    values[off] = value;
                    return previous;
                } else if (compare(searchKey, key)) {
                    return values[off];
                }
            }
            resize(this.bits + 1);
        }
    }

    @Nullable
    public V get(@Nonnull final Object key) {
        int off = getBucketOffset(key);
        final int end = sweep + off;
        for (; off < end; off++) {
            if (keys[off] != null && compare(keys[off], key)) {
                return values[off];
            }
        }
        return null;
    }

    @Nullable
    public V remove(@Nonnull final Object key) {
        int off = getBucketOffset(key);
        final int end = sweep + off;
        for (; off < end; off++) {
            if (keys[off] != null && compare(keys[off], key)) {
                keys[off] = null;
                V previous = values[off];
                values[off] = null;
                size--;
                return previous;
            }
        }
        return null;
    }

    public int size() {
        return size;
    }

    public void putAll(@Nonnull final Map<? extends K, ? extends V> m) {
        for (K key : m.keySet()) {
            put(key, m.get(key));
        }
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public boolean containsKey(@Nonnull final Object key) {
        return get(key) != null;
    }

    public boolean containsValue(@Nonnull final Object value) {
        for (V v : values) {
            if (v != null && compare(v, value)) {
                return true;
            }
        }
        return false;
    }

    public void clear() {
        Arrays.fill(keys, null);
        Arrays.fill(values, null);
        this.size = 0;
    }

    @Nonnull
    public Set<K> keySet() {
        final Set<K> set = new HashSet<K>();
        for (K key : keys) {
            if (key != null) {
                set.add(key);
            }
        }
        return set;
    }

    @Nonnull
    public Collection<V> values() {
        final Collection<V> list = new ArrayList<V>();
        for (V value : values) {
            if (value != null) {
                list.add(value);
            }
        }
        return list;
    }

    @Nonnull
    public Set<Entry<K, V>> entrySet() {
        final Set<Entry<K, V>> set = new HashSet<Entry<K, V>>();
        for (K key : keys) {
            if (key != null) {
                set.add(new MapEntry<K, V>(this, key));
            }
        }
        return set;
    }

    private static final class MapEntry<K, V> implements Map.Entry<K, V> {
        private final Map<K, V> map;
        private final K key;

        public MapEntry(Map<K, V> map, K key) {
            this.map = map;
            this.key = key;
        }

        @Override
        public K getKey() {
            return key;
        }

        @Override
        public V getValue() {
            return map.get(key);
        }

        @Override
        public V setValue(V value) {
            return map.put(key, value);
        }
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        // remember the number of bits
        out.writeInt(this.bits);
        // remember the total number of entries
        out.writeInt(this.size);
        // write all entries
        for (int x = 0; x < this.keys.length; x++) {
            if (keys[x] != null) {
                out.writeObject(keys[x]);
                out.writeObject(values[x]);
            }
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        // resize to old bit size
        int bitSize = in.readInt();
        if (bitSize != bits) {
            resize(bitSize);
        }
        // read all entries
        int size = in.readInt();
        for (int x = 0; x < size; x++) {
            this.put((K) in.readObject(), (V) in.readObject());
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + ' ' + size;
    }

    @SuppressWarnings("unchecked")
    private void resize(final int bits) {
        this.bits = bits;
        this.sweepbits = bits / 4;
        this.sweep = MathUtils.powerOf(2, sweepbits) * 4;
        this.sweepmask = MathUtils.bitMask(bits - sweepbits) << sweepbits;

        // remember old values so we can recreate the entries
        final K[] existingKeys = this.keys;
        final V[] existingValues = this.values;

        // create the arrays
        this.values = (V[]) new Object[MathUtils.powerOf(2, bits) + sweep];
        this.keys = (K[]) new Object[values.length];
        this.size = 0;

        // re-add the previous entries if resizing
        if (existingKeys != null) {
            for (int x = 0; x < existingKeys.length; x++) {
                final K k = existingKeys[x];
                if (k != null) {
                    put(k, existingValues[x]);
                }
            }
        }
    }

    private int getBucketOffset(@Nonnull final Object key) {
        return (key.hashCode() << sweepbits) & sweepmask;
    }

    private static boolean compare(@Nonnull final Object v1, @Nonnull final Object v2) {
        return v1 == v2 || v1.equals(v2);
    }

    public IMapIterator<K, V> entries() {
        return new MapIterator(false);
    }

    public IMapIterator<K, V> entries(boolean releaseSeen) {
        return new MapIterator(releaseSeen);
    }

    private final class MapIterator implements IMapIterator<K, V> {

        final boolean releaseSeen;
        int nextEntry;
        int lastEntry = -1;

        MapIterator(boolean releaseSeen) {
            this.releaseSeen = releaseSeen;
            this.nextEntry = nextEntry(0);
        }

        /** find the index of next full entry */
        int nextEntry(int index) {
            while (index < keys.length && keys[index] == null) {
                index++;
            }
            return index;
        }

        @Override
        public boolean hasNext() {
            return nextEntry < keys.length;
        }

        @Override
        public int next() {
            if (releaseSeen) {
                free(lastEntry);
            }
            if (!hasNext()) {
                return -1;
            }
            int curEntry = nextEntry;
            this.lastEntry = curEntry;
            this.nextEntry = nextEntry(curEntry + 1);
            return curEntry;
        }

        @Override
        public K getKey() {
            return keys[lastEntry];
        }

        @Override
        public V getValue() {
            return values[lastEntry];
        }

        @Override
        public <T extends Copyable<V>> void getValue(T probe) {
            probe.copyFrom(getValue());
        }

        private void free(int index) {
            if (index >= 0) {
                keys[index] = null;
                values[index] = null;
            }
        }

    }
}
