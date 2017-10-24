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
package hivemall.utils.collections.maps;

import hivemall.utils.collections.IMapIterator;
import hivemall.utils.lang.Copyable;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.Primes;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * An open-addressing hash table using double-hashing.
 *
 * <pre>
 * Primary hash function: h1(k) = k mod m
 * Secondary hash function: h2(k) = 1 + (k mod(m-2))
 * </pre>
 *
 * @see http://en.wikipedia.org/wiki/Double_hashing
 */
public final class OpenHashTable<K, V> implements Externalizable {

    public static final float DEFAULT_LOAD_FACTOR = 0.75f;
    public static final float DEFAULT_GROW_FACTOR = 2.0f;

    protected static final byte FREE = 0;
    protected static final byte FULL = 1;
    protected static final byte REMOVED = 2;

    protected/* final */float _loadFactor;
    protected/* final */float _growFactor;

    protected int _used = 0;
    protected int _threshold;

    protected K[] _keys;
    protected V[] _values;
    protected byte[] _states;

    /**
     * Only for {@link Externalizable}
     */
    public OpenHashTable() {}

    public OpenHashTable(int size) {
        this(size, DEFAULT_LOAD_FACTOR, DEFAULT_GROW_FACTOR);
    }

    @SuppressWarnings("unchecked")
    public OpenHashTable(int size, float loadFactor, float growFactor) {
        if (size < 1) {
            throw new IllegalArgumentException();
        }
        this._loadFactor = loadFactor;
        this._growFactor = growFactor;
        int actualSize = Primes.findLeastPrimeNumber(size);
        this._keys = (K[]) new Object[actualSize];
        this._values = (V[]) new Object[actualSize];
        this._states = new byte[actualSize];
        this._threshold = Math.round(actualSize * _loadFactor);
    }

    public OpenHashTable(@Nonnull K[] keys, @Nonnull V[] values, @Nonnull byte[] states, int used) {
        this._used = used;
        this._threshold = keys.length;
        this._keys = keys;
        this._values = values;
        this._states = states;
    }

    public Object[] getKeys() {
        return _keys;
    }

    public Object[] getValues() {
        return _values;
    }

    public byte[] getStates() {
        return _states;
    }

    public boolean containsKey(@CheckForNull final K key) {
        return findKey(key) >= 0;
    }

    public V get(@CheckForNull final K key) {
        final int i = findKey(key);
        if (i < 0) {
            return null;
        }
        return _values[i];
    }

    public V put(@CheckForNull final K key, @Nullable final V value) {
        Preconditions.checkNotNull(key);

        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final K[] keys = _keys;
        final V[] values = _values;
        final byte[] states = _states;

        if (states[keyIdx] == FULL) {// double hashing
            if (equals(keys[keyIdx], key)) {
                V old = values[keyIdx];
                values[keyIdx] = value;
                return old;
            }
            // try second hash
            final int loopIndex = keyIdx;
            final int decr = 1 + (hash % (keyLength - 2));
            for (;;) {
                keyIdx -= decr;
                if (keyIdx < 0) {
                    keyIdx += keyLength;
                }
                if (keyIdx == loopIndex) {
                    throw new IllegalStateException(
                        "Detected infinite loop where key=" + key + ", keyIdx=" + keyIdx);
                }

                final byte state = states[keyIdx];
                if (state == FREE) {
                    break;
                }
                if (equals(keys[keyIdx], key)) {
                    if (states[keyIdx] == FULL) {
                        V old = values[keyIdx];
                        values[keyIdx] = value;
                        return old;
                    } else {
                        assert (state == REMOVED);
                        break;
                    }
                }
            }
        }
        keys[keyIdx] = key;
        values[keyIdx] = value;
        states[keyIdx] = FULL;
        ++_used;
        return null;
    }

    private static boolean equals(@Nonnull final Object k1, @Nonnull final Object k2) {
        return k1 == k2 || k1.equals(k2);
    }

    /** @return expanded or not */
    protected boolean preAddEntry(int index) {
        if ((_used + 1) >= _threshold) {// filled enough
            int newCapacity = Math.round(_keys.length * _growFactor);
            ensureCapacity(newCapacity);
            return true;
        }
        return false;
    }

    protected int findKey(@CheckForNull final K key) {
        Preconditions.checkNotNull(key);

        final K[] keys = _keys;
        final byte[] states = _states;
        final int keyLength = keys.length;

        // double hashing
        final int hash = keyHash(key);
        final int decr = 1 + (hash % (keyLength - 2));
        final int startIndex = hash % keyLength;
        for (int keyIdx = startIndex;;) {
            final byte state = states[keyIdx];
            if (state == FREE) {
                return -1;
            }
            if (equals(keys[keyIdx], key)) {
                if (state == FULL) {
                    return keyIdx;
                } else {
                    assert (state == REMOVED);
                    return -1;
                }
            }
            keyIdx -= decr;
            if (keyIdx < 0) {
                keyIdx += keyLength;
            }
            if (keyIdx == startIndex) {
                throw new IllegalStateException(
                    "Detected infinite loop where key=" + key + ", keyIdx=" + keyIdx);
            }
        }
    }

    public V remove(@CheckForNull final K key) {
        final int keyIdx = findKey(key);
        if (keyIdx == -1) {
            return null;
        }

        V old = _values[keyIdx];
        _states[keyIdx] = REMOVED;
        --_used;
        return old;
    }

    public int size() {
        return _used;
    }

    public void clear() {
        Arrays.fill(_states, FREE);
        this._used = 0;
    }

    public IMapIterator<K, V> entries() {
        return new MapIterator(false);
    }

    public IMapIterator<K, V> entries(boolean releaseSeen) {
        return new MapIterator(releaseSeen);
    }

    @Override
    public String toString() {
        int len = size() * 10 + 2;
        final StringBuilder buf = new StringBuilder(len);
        buf.append('{');
        final IMapIterator<K, V> i = entries();
        while (i.next() != -1) {
            String key = i.getKey().toString();
            buf.append(key);
            buf.append('=');
            buf.append(i.getValue());
            if (i.hasNext()) {
                buf.append(',');
            }
        }
        buf.append('}');
        return buf.toString();
    }

    protected void ensureCapacity(@Nonnegative int newCapacity) {
        int prime = Primes.findLeastPrimeNumber(newCapacity);
        rehash(prime);
        this._threshold = Math.round(prime * _loadFactor);
    }

    @SuppressWarnings("unchecked")
    private void rehash(@Nonnegative final int newCapacity) {
        final K[] oldKeys = _keys;
        final V[] oldValues = _values;
        final byte[] oldStates = _states;

        final int oldCapacity = oldKeys.length;
        if (newCapacity <= oldCapacity) {
            throw new IllegalArgumentException("new: " + newCapacity + ", old: " + oldCapacity);
        }

        final K[] newkeys = (K[]) new Object[newCapacity];
        final V[] newValues = (V[]) new Object[newCapacity];
        final byte[] newStates = new byte[newCapacity];
        int used = 0;
        for (int i = 0; i < oldCapacity; i++) {
            if (oldStates[i] == FULL) {
                used++;
                final K k = oldKeys[i];
                final V v = oldValues[i];
                final int hash = keyHash(k);
                int keyIdx = hash % newCapacity;
                if (newStates[keyIdx] == FULL) {// second hashing
                    final int decr = 1 + (hash % (newCapacity - 2));
                    final int loopIndex = keyIdx;
                    while (newStates[keyIdx] != FREE) {
                        keyIdx -= decr;
                        if (keyIdx < 0) {
                            keyIdx += newCapacity;
                        }
                        if (keyIdx == loopIndex) {
                            throw new IllegalStateException(
                                "Detected infinite loop where key=" + k + ", keyIdx=" + keyIdx);
                        }
                    }
                }
                newkeys[keyIdx] = k;
                newValues[keyIdx] = v;
                newStates[keyIdx] = FULL;
            }
        }
        this._keys = newkeys;
        this._values = newValues;
        this._states = newStates;
        this._used = used;
    }

    private static int keyHash(@Nonnull final Object key) {
        int hash = key.hashCode();
        return hash & 0x7fffffff;
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
            while (index < _keys.length && _states[index] != FULL) {
                index++;
            }
            return index;
        }

        public boolean hasNext() {
            return nextEntry < _keys.length;
        }

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

        public K getKey() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _keys[lastEntry];
        }

        public V getValue() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _values[lastEntry];
        }

        @Override
        public <T extends Copyable<V>> void getValue(T probe) {
            probe.copyFrom(getValue());
        }

        private void free(int index) {
            if (index < 0) {
                return; // should not happen
            }
            _keys[index] = null;
            _values[index] = null;
            _states[index] = FREE;
        }
    }

    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeFloat(_loadFactor);
        out.writeFloat(_growFactor);
        out.writeInt(_used);

        final int size = _keys.length;
        out.writeInt(size);

        for (int i = 0; i < size; i++) {
            out.writeObject(_keys[i]);
            out.writeObject(_values[i]);
            out.writeByte(_states[i]);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        this._loadFactor = in.readFloat();
        this._growFactor = in.readFloat();
        this._used = in.readInt();

        final int size = in.readInt();
        final Object[] keys = new Object[size];
        final Object[] values = new Object[size];
        final byte[] states = new byte[size];
        for (int i = 0; i < size; i++) {
            keys[i] = in.readObject();
            values[i] = in.readObject();
            states[i] = in.readByte();
        }
        this._threshold = size;
        this._keys = (K[]) keys;
        this._values = (V[]) values;
        this._states = states;
    }

}
