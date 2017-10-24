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
public final class Long2FloatOpenHashTable implements Externalizable {

    protected static final byte FREE = 0;
    protected static final byte FULL = 1;
    protected static final byte REMOVED = 2;

    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    private static final float DEFAULT_GROW_FACTOR = 2.0f;

    protected final transient float _loadFactor;
    protected final transient float _growFactor;

    protected int _used = 0;
    protected int _threshold;
    protected float defaultReturnValue = 0.f;

    protected long[] _keys;
    protected float[] _values;
    protected byte[] _states;

    protected Long2FloatOpenHashTable(int size, float loadFactor, float growFactor,
            boolean forcePrime) {
        if (size < 1) {
            throw new IllegalArgumentException();
        }
        this._loadFactor = loadFactor;
        this._growFactor = growFactor;
        int actualSize = forcePrime ? Primes.findLeastPrimeNumber(size) : size;
        this._keys = new long[actualSize];
        this._values = new float[actualSize];
        this._states = new byte[actualSize];
        this._threshold = (int) (actualSize * _loadFactor);
    }

    public Long2FloatOpenHashTable(int size, int loadFactor, int growFactor) {
        this(size, loadFactor, growFactor, true);
    }

    public Long2FloatOpenHashTable(int size) {
        this(size, DEFAULT_LOAD_FACTOR, DEFAULT_GROW_FACTOR, true);
    }

    /**
     * Only for {@link Externalizable}
     */
    public Long2FloatOpenHashTable() {
        this._loadFactor = DEFAULT_LOAD_FACTOR;
        this._growFactor = DEFAULT_GROW_FACTOR;
    }

    public void defaultReturnValue(float v) {
        this.defaultReturnValue = v;
    }

    public boolean containsKey(final long key) {
        return _findKey(key) >= 0;
    }

    /**
     * @return defaultReturnValue if not found
     */
    public float get(final long key) {
        return get(key, defaultReturnValue);
    }

    public float get(final long key, final float defaultValue) {
        final int i = _findKey(key);
        if (i < 0) {
            return defaultValue;
        }
        return _values[i];
    }

    public float _get(final int index) {
        if (index < 0) {
            return defaultReturnValue;
        }
        return _values[index];
    }

    public float _set(final int index, final float value) {
        float old = _values[index];
        _values[index] = value;
        return old;
    }

    public float _remove(final int index) {
        _states[index] = REMOVED;
        --_used;
        return _values[index];
    }

    public float put(final long key, final float value) {
        return put(key, value, defaultReturnValue);
    }

    public float put(final long key, final float value, final float defaultValue) {
        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final long[] keys = _keys;
        final float[] values = _values;
        final byte[] states = _states;

        if (states[keyIdx] == FULL) {// double hashing
            if (keys[keyIdx] == key) {
                float old = values[keyIdx];
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
                if (keys[keyIdx] == key) {
                    if (state == FULL) {
                        float old = values[keyIdx];
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
        return defaultValue;
    }

    /** Return weather the required slot is free for new entry */
    protected boolean isFree(final int index, final long key) {
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

    /**
     * @return -1 if not found
     */
    public int _findKey(final long key) {
        final long[] keys = _keys;
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
            if (keys[keyIdx] == key) {
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

    public float remove(final long key) {
        final int keyIdx = _findKey(key);
        if (keyIdx == -1) {
            return defaultReturnValue;
        }

        float old = _values[keyIdx];
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

    public IMapIterator entries() {
        return new MapIterator();
    }

    @Override
    public String toString() {
        int len = size() * 10 + 2;
        StringBuilder buf = new StringBuilder(len);
        buf.append('{');
        IMapIterator i = entries();
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

    protected void ensureCapacity(final int newCapacity) {
        int prime = Primes.findLeastPrimeNumber(newCapacity);
        rehash(prime);
        this._threshold = Math.round(prime * _loadFactor);
    }

    private void rehash(final int newCapacity) {
        final long[] oldKeys = _keys;
        final float[] oldValues = _values;
        final byte[] oldStates = _states;

        final int oldCapacity = _keys.length;
        if (newCapacity <= oldCapacity) {
            throw new IllegalArgumentException("new: " + newCapacity + ", old: " + oldCapacity);
        }

        final long[] newkeys = new long[newCapacity];
        final float[] newValues = new float[newCapacity];
        final byte[] newStates = new byte[newCapacity];
        int used = 0;
        for (int i = 0; i < oldCapacity; i++) {
            if (oldStates[i] == FULL) {
                used++;
                final long k = oldKeys[i];
                final float v = oldValues[i];
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

    private static int keyHash(final long key) {
        return (int) (key ^ (key >>> 32)) & 0x7FFFFFFF;
    }

    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(_threshold);
        out.writeInt(_used);

        out.writeInt(_keys.length);
        IMapIterator i = entries();
        while (i.next() != -1) {
            out.writeLong(i.getKey());
            out.writeFloat(i.getValue());
        }
    }

    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        this._threshold = in.readInt();
        this._used = in.readInt();

        final int keylen = in.readInt();
        final long[] keys = new long[keylen];
        final float[] values = new float[keylen];
        final byte[] states = new byte[keylen];
        for (int i = 0; i < _used; i++) {
            final long k = in.readLong();
            final float v = in.readFloat();
            final int hash = keyHash(k);
            int keyIdx = hash % keylen;
            if (states[keyIdx] != FREE) {// second hash
                final int decr = 1 + (hash % (keylen - 2));
                for (;;) {
                    keyIdx -= decr;
                    if (keyIdx < 0) {
                        keyIdx += keylen;
                    }
                    if (states[keyIdx] == FREE) {
                        break;
                    }
                }
            }
            states[keyIdx] = FULL;
            keys[keyIdx] = k;
            values[keyIdx] = v;
        }
        this._keys = keys;
        this._values = values;
        this._states = states;
    }

    public interface IMapIterator {

        public boolean hasNext();

        /**
         * @return -1 if not found
         */
        public int next();

        public long getKey();

        public float getValue();

    }

    private final class MapIterator implements IMapIterator {

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

        public long getKey() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _keys[lastEntry];
        }

        public float getValue() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _values[lastEntry];
        }
    }
}
