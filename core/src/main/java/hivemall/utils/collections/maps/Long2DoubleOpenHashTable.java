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
public final class Long2DoubleOpenHashTable implements Externalizable {

    protected static final byte FREE = 0;
    protected static final byte FULL = 1;
    protected static final byte REMOVED = 2;

    private static final float DEFAULT_LOAD_FACTOR = 0.75f;
    private static final float DEFAULT_GROW_FACTOR = 2.0f;

    private static final float SHRINK_FACTOR = 0.1f; // at least 10% of table must be FREE
    private static final float GROW_FACTOR_AT_SHRINK = 1.7f;

    protected/* final */float _loadFactor;
    protected/* final */float _growFactor;

    protected int _used;
    protected int _freeEntries;

    /** Used entry threshold to grow table */
    protected int _growThreshold;
    /**
     * Free entry threshold to shrink table. Shrink threshold will be set in the first expansion to
     * avoid shrink at very early remove().
     */
    protected int _shrinkThreshold;

    protected double _defaultReturnValue = 0.d;

    protected long[] _keys;
    protected double[] _values;
    protected byte[] _states;

    protected Long2DoubleOpenHashTable(int size, float loadFactor, float growFactor,
            boolean forcePrime) {
        if (size < 1) {
            throw new IllegalArgumentException();
        }
        this._loadFactor = loadFactor;
        this._growFactor = growFactor;
        int actualSize = forcePrime ? Primes.findLeastPrimeNumber(size) : size;
        this._keys = new long[actualSize];
        this._values = new double[actualSize];
        this._states = new byte[actualSize];
        this._used = 0;
        this._freeEntries = actualSize;
        this._growThreshold = Math.round(actualSize * _loadFactor);
        this._shrinkThreshold = Math.round(actualSize * SHRINK_FACTOR);
    }

    public Long2DoubleOpenHashTable(int size, int loadFactor, int growFactor) {
        this(size, loadFactor, growFactor, true);
    }

    public Long2DoubleOpenHashTable(int size) {
        this(size, DEFAULT_LOAD_FACTOR, DEFAULT_GROW_FACTOR, true);
    }

    /**
     * Only for {@link Externalizable}
     */
    public Long2DoubleOpenHashTable() {// required for serialization
        this._loadFactor = DEFAULT_LOAD_FACTOR;
        this._growFactor = DEFAULT_GROW_FACTOR;
    }

    public void defaultReturnValue(double v) {
        this._defaultReturnValue = v;
    }

    public boolean containsKey(final long key) {
        return _findKey(key) >= 0;
    }

    /**
     * @return defaultReturnValue if not found
     */
    public double get(final long key) {
        return get(key, _defaultReturnValue);
    }

    public double get(final long key, final double defaultValue) {
        final int i = _findKey(key);
        if (i == -1) {
            return defaultValue;
        }
        return _values[i];
    }

    public double _get(final int index) {
        if (index < 0) {
            return _defaultReturnValue;
        }
        return _values[index];
    }

    public double _set(final int index, final double value) {
        double old = _values[index];
        _values[index] = value;
        return old;
    }

    public double _remove(final int index) {
        _states[index] = REMOVED;
        --_used;
        return _values[index];
    }

    public double put(final long key, final double value) {
        return put(key, value, _defaultReturnValue);
    }

    public double put(final long key, final double value, final double defaultValue) {
        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final long[] keys = _keys;
        final double[] values = _values;
        final byte[] states = _states;

        byte state = states[keyIdx];
        if (state == FULL) {// double hashing
            if (keys[keyIdx] == key) {
                double old = values[keyIdx];
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

                state = states[keyIdx];
                if (state == FREE) {
                    break;
                }
                if (keys[keyIdx] == key) {
                    if (state == FULL) {
                        double old = values[keyIdx];
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

        if (state == FREE) {
            _freeEntries--;
            if (_freeEntries < _shrinkThreshold) {
                int newCapacity = Math.max(keys.length, Math.round(_used * GROW_FACTOR_AT_SHRINK));
                ensureCapacity(newCapacity);
            }
        }

        return defaultValue;
    }

    /** @return expanded or not */
    protected boolean preAddEntry(final int index) {
        if ((_used + 1) >= _growThreshold) {// too filled
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

    public double remove(final long key) {
        final int keyIdx = _findKey(key);
        if (keyIdx == -1) {
            return _defaultReturnValue;
        }

        double old = _values[keyIdx];
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
        this._freeEntries = _states.length;
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
    }

    private void rehash(final int newCapacity) {
        final long[] oldKeys = _keys;
        final double[] oldValues = _values;
        final byte[] oldStates = _states;
        final int oldCapacity = _keys.length;

        final long[] newkeys = new long[newCapacity];
        final double[] newValues = new double[newCapacity];
        final byte[] newStates = new byte[newCapacity];
        int used = 0;
        for (int i = 0; i < oldCapacity; i++) {
            if (oldStates[i] != FULL) {
                continue;
            }
            final long k = oldKeys[i];
            final double v = oldValues[i];
            final int hash = keyHash(k);
            int keyIdx = hash % newCapacity;
            if (newStates[keyIdx] == FULL) {// second hashing
                final int decr = 1 + (hash % (newCapacity - 2));
                final int loopIndex = keyIdx;
                do {
                    keyIdx -= decr;
                    if (keyIdx < 0) {
                        keyIdx += newCapacity;
                    }
                    if (keyIdx == loopIndex) {
                        throw new IllegalStateException(
                            "Detected infinite loop where key=" + k + ", keyIdx=" + keyIdx);
                    }
                } while (newStates[keyIdx] != FREE);
            }
            newkeys[keyIdx] = k;
            newValues[keyIdx] = v;
            newStates[keyIdx] = FULL;
            used++;
        }
        this._keys = newkeys;
        this._values = newValues;
        this._states = newStates;
        this._used = used;
        this._freeEntries = newCapacity - used;
        this._growThreshold = Math.round(newCapacity * _loadFactor);
        this._shrinkThreshold = Math.round(newCapacity * SHRINK_FACTOR);
    }

    private static int keyHash(final long key) {
        return (int) (key ^ (key >>> 32)) & 0x7FFFFFFF;
    }

    public interface IMapIterator {

        public boolean hasNext();

        /**
         * @return -1 if not found
         */
        public int next();

        public long getKey();

        public double getValue();

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

        public double getValue() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _values[lastEntry];
        }
    }

    @Override
    public void writeExternal(@Nonnull ObjectOutput out) throws IOException {
        out.writeFloat(_loadFactor);
        out.writeFloat(_growFactor);
        out.writeInt(_used);
        out.writeDouble(_defaultReturnValue);

        final IMapIterator i = entries();
        while (i.next() != -1) {
            out.writeLong(i.getKey());
            out.writeDouble(i.getValue());
        }
    }

    @Override
    public void readExternal(@Nonnull ObjectInput in) throws IOException, ClassNotFoundException {
        this._loadFactor = in.readFloat();
        this._growFactor = in.readFloat();
        final int used = in.readInt();
        this._defaultReturnValue = in.readDouble();

        final int newCapacity = Primes.findLeastPrimeNumber(Math.round(used * 1.7f));
        final long[] keys = new long[newCapacity];
        final double[] values = new double[newCapacity];
        final byte[] states = new byte[newCapacity];
        for (int i = 0; i < used; i++) {
            final long k = in.readLong();
            final double v = in.readDouble();
            final int hash = keyHash(k);
            int keyIdx = hash % newCapacity;
            if (states[keyIdx] != FREE) {// second hashing
                final int decr = 1 + (hash % (newCapacity - 2));
                final int loopIndex = keyIdx;
                do {
                    keyIdx -= decr;
                    if (keyIdx < 0) {
                        keyIdx += newCapacity;
                    }
                    if (keyIdx == loopIndex) {
                        throw new IllegalStateException(
                            "Detected infinite loop where key=" + k + ", keyIdx=" + keyIdx);
                    }
                } while (states[keyIdx] != FREE);
            }
            keys[keyIdx] = k;
            values[keyIdx] = v;
            states[keyIdx] = FULL;
        }
        this._keys = keys;
        this._values = values;
        this._states = states;
        this._used = used;
        this._freeEntries = newCapacity - used;
        this._growThreshold = Math.round(newCapacity * _loadFactor);
        this._shrinkThreshold = Math.round(newCapacity * SHRINK_FACTOR);
    }

}
