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

import hivemall.utils.math.Primes;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import javax.annotation.Nonnull;

/**
 * An open-addressing hash table using double hashing.
 *
 * <pre>
 * Primary hash function: h1(k) = k mod m
 * Secondary hash function: h2(k) = 1 + (k mod(m-2))
 * </pre>
 *
 * @see http://en.wikipedia.org/wiki/Double_hashing
 */
public final class IntOpenHashTable<V> implements Externalizable {
    private static final long serialVersionUID = -8162355845665353513L;

    public static final float DEFAULT_LOAD_FACTOR = 0.75f;
    public static final float DEFAULT_GROW_FACTOR = 2.0f;

    protected static final byte FREE = 0;
    protected static final byte FULL = 1;
    protected static final byte REMOVED = 2;

    protected/* final */float _loadFactor;
    protected/* final */float _growFactor;

    protected int _used;
    protected int _threshold;

    protected int[] _keys;
    protected V[] _values;
    protected byte[] _states;

    public IntOpenHashTable() {} // for Externalizable    

    public IntOpenHashTable(int size) {
        this(size, DEFAULT_LOAD_FACTOR, DEFAULT_GROW_FACTOR, true);
    }

    public IntOpenHashTable(int size, float loadFactor, float growFactor) {
        this(size, loadFactor, growFactor, true);
    }

    @SuppressWarnings("unchecked")
    protected IntOpenHashTable(int size, float loadFactor, float growFactor, boolean forcePrime) {
        if (size < 1) {
            throw new IllegalArgumentException();
        }
        this._loadFactor = loadFactor;
        this._growFactor = growFactor;
        this._used = 0;
        int actualSize = forcePrime ? Primes.findLeastPrimeNumber(size) : size;
        this._threshold = Math.round(actualSize * _loadFactor);
        this._keys = new int[actualSize];
        this._values = (V[]) new Object[actualSize];
        this._states = new byte[actualSize];
    }

    public IntOpenHashTable(@Nonnull int[] keys, @Nonnull V[] values, @Nonnull byte[] states,
            int used) {
        this._loadFactor = DEFAULT_LOAD_FACTOR;
        this._growFactor = DEFAULT_GROW_FACTOR;
        this._used = used;
        this._threshold = keys.length;
        this._keys = keys;
        this._values = values;
        this._states = states;
    }

    @Nonnull
    public int[] getKeys() {
        return _keys;
    }

    @Nonnull
    public Object[] getValues() {
        return _values;
    }

    @Nonnull
    public byte[] getStates() {
        return _states;
    }

    public boolean containsKey(final int key) {
        return findKey(key) >= 0;
    }

    public V get(final int key) {
        final int i = findKey(key);
        if (i < 0) {
            return null;
        }
        return _values[i];
    }

    public V put(final int key, final V value) {
        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        final boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final int[] keys = _keys;
        final V[] values = _values;
        final byte[] states = _states;

        if (states[keyIdx] == FULL) {// double hashing
            if (keys[keyIdx] == key) {
                V old = values[keyIdx];
                values[keyIdx] = value;
                return old;
            }
            // try second hash
            final int decr = 1 + (hash % (keyLength - 2));
            for (;;) {
                keyIdx -= decr;
                if (keyIdx < 0) {
                    keyIdx += keyLength;
                }
                if (isFree(keyIdx, key)) {
                    break;
                }
                if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                    V old = values[keyIdx];
                    values[keyIdx] = value;
                    return old;
                }
            }
        }
        keys[keyIdx] = key;
        values[keyIdx] = value;
        states[keyIdx] = FULL;
        ++_used;
        return null;
    }

    public V putIfAbsent(final int key, final V value) {
        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        final boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final int[] keys = _keys;
        final V[] values = _values;
        final byte[] states = _states;

        if (states[keyIdx] == FULL) {// second hashing
            if (keys[keyIdx] == key) {
                return values[keyIdx];
            }
            // try second hash
            final int decr = 1 + (hash % (keyLength - 2));
            for (;;) {
                keyIdx -= decr;
                if (keyIdx < 0) {
                    keyIdx += keyLength;
                }
                if (isFree(keyIdx, key)) {
                    break;
                }
                if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                    return values[keyIdx];
                }
            }
        }
        keys[keyIdx] = key;
        values[keyIdx] = value;
        states[keyIdx] = FULL;
        _used++;
        return null;
    }

    /** Return weather the required slot is free for new entry */
    protected boolean isFree(final int index, final int key) {
        final byte stat = _states[index];
        if (stat == FREE) {
            return true;
        }
        if (stat == REMOVED && _keys[index] == key) {
            return true;
        }
        return false;
    }

    /** @return expanded or not */
    protected boolean preAddEntry(final int index) {
        if ((_used + 1) >= _threshold) {// too filled
            int newCapacity = Math.round(_keys.length * _growFactor);
            ensureCapacity(newCapacity);
            return true;
        }
        return false;
    }

    private int findKey(final int key) {
        final int[] keys = _keys;
        final byte[] states = _states;
        final int keyLength = keys.length;

        final int hash = keyHash(key);
        int keyIdx = hash % keyLength;
        if (states[keyIdx] != FREE) {
            if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                return keyIdx;
            }
            // try second hash
            final int decr = 1 + (hash % (keyLength - 2));
            for (;;) {
                keyIdx -= decr;
                if (keyIdx < 0) {
                    keyIdx += keyLength;
                }
                if (isFree(keyIdx, key)) {
                    return -1;
                }
                if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                    return keyIdx;
                }
            }
        }
        return -1;
    }

    public V remove(final int key) {
        final int[] keys = _keys;
        final V[] values = _values;
        final byte[] states = _states;
        final int keyLength = keys.length;

        final int hash = keyHash(key);
        int keyIdx = hash % keyLength;
        if (states[keyIdx] != FREE) {
            if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                V old = values[keyIdx];
                states[keyIdx] = REMOVED;
                --_used;
                return old;
            }
            //  second hash
            final int decr = 1 + (hash % (keyLength - 2));
            for (;;) {
                keyIdx -= decr;
                if (keyIdx < 0) {
                    keyIdx += keyLength;
                }
                if (states[keyIdx] == FREE) {
                    return null;
                }
                if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                    V old = values[keyIdx];
                    states[keyIdx] = REMOVED;
                    --_used;
                    return old;
                }
            }
        }
        return null;
    }

    @Nonnull
    public IMapIterator<V> entries() {
        return new MapIterator();
    }

    public int size() {
        return _used;
    }

    public int capacity() {
        return _keys.length;
    }

    public void clear() {
        Arrays.fill(_states, FREE);
        this._used = 0;
    }

    @Override
    public String toString() {
        int len = size() * 10 + 2;
        final StringBuilder buf = new StringBuilder(len);
        buf.append('{');
        final IMapIterator<V> i = entries();
        while (i.next() != -1) {
            buf.append(i.getKey());
            buf.append('=');
            buf.append(i.getValue());
            if (i.hasNext()) {
                buf.append(',');
            }
        }
        buf.append('}');
        return buf.toString();
    }

    private void ensureCapacity(final int newCapacity) {
        int prime = Primes.findLeastPrimeNumber(newCapacity);
        rehash(prime);
        this._threshold = Math.round(prime * _loadFactor);
    }

    @SuppressWarnings("unchecked")
    private void rehash(final int newCapacity) {
        int oldCapacity = _keys.length;
        if (newCapacity <= oldCapacity) {
            throw new IllegalArgumentException("new: " + newCapacity + ", old: " + oldCapacity);
        }
        final int[] oldKeys = _keys;
        final V[] oldValues = _values;
        final byte[] oldStates = _states;
        final int[] newkeys = new int[newCapacity];
        final V[] newValues = (V[]) new Object[newCapacity];
        final byte[] newStates = new byte[newCapacity];
        int used = 0;
        for (int i = 0; i < oldCapacity; i++) {
            if (oldStates[i] == FULL) {
                used++;
                final int k = oldKeys[i];
                final V v = oldValues[i];
                final int hash = keyHash(k);
                int keyIdx = hash % newCapacity;
                if (newStates[keyIdx] == FULL) {// second hashing
                    int decr = 1 + (hash % (newCapacity - 2));
                    while (newStates[keyIdx] != FREE) {
                        keyIdx -= decr;
                        if (keyIdx < 0) {
                            keyIdx += newCapacity;
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

    private static int keyHash(final int key) {
        return key & 0x7fffffff;
    }

    @Override
    public void writeExternal(@Nonnull final ObjectOutput out) throws IOException {
        out.writeFloat(_loadFactor);
        out.writeFloat(_growFactor);
        out.writeInt(_used);

        final int size = _keys.length;
        out.writeInt(size);

        for (int i = 0; i < size; i++) {
            out.writeInt(_keys[i]);
            out.writeObject(_values[i]);
            out.writeByte(_states[i]);
        }
    }

    @SuppressWarnings("unchecked")
    public void readExternal(@Nonnull final ObjectInput in) throws IOException,
            ClassNotFoundException {
        this._loadFactor = in.readFloat();
        this._growFactor = in.readFloat();
        this._used = in.readInt();

        final int size = in.readInt();
        final int[] keys = new int[size];
        final Object[] values = new Object[size];
        final byte[] states = new byte[size];
        for (int i = 0; i < size; i++) {
            keys[i] = in.readInt();
            values[i] = in.readObject();
            states[i] = in.readByte();
        }
        this._threshold = size;
        this._keys = keys;
        this._values = (V[]) values;
        this._states = states;
    }

    public interface IMapIterator<V> {

        public boolean hasNext();

        /**
         * @return -1 if not found
         */
        public int next();

        public int getKey();

        public V getValue();

    }

    private final class MapIterator implements IMapIterator<V> {

        int nextEntry;
        int lastEntry = -1;

        MapIterator() {
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
            if (!hasNext()) {
                return -1;
            }
            int curEntry = nextEntry;
            this.lastEntry = curEntry;
            this.nextEntry = nextEntry(curEntry + 1);
            return curEntry;
        }

        public int getKey() {
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
    }

}
