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

import hivemall.utils.codec.VariableByteCodec;
import hivemall.utils.codec.ZigZagLEB128Codec;
import hivemall.utils.math.Primes;

import java.io.DataInput;
import java.io.DataOutput;
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
public class Int2LongOpenHashTable implements Externalizable {

    protected static final byte FREE = 0;
    protected static final byte FULL = 1;
    protected static final byte REMOVED = 2;

    public static final int DEFAULT_SIZE = 65536;
    public static final float DEFAULT_LOAD_FACTOR = 0.75f;
    public static final float DEFAULT_GROW_FACTOR = 2.0f;

    protected final transient float _loadFactor;
    protected final transient float _growFactor;

    protected int[] _keys;
    protected long[] _values;
    protected byte[] _states;

    protected int _used;
    protected int _threshold;
    protected long defaultReturnValue = -1L;

    /**
     * Constructor for Externalizable. Should not be called otherwise.
     */
    public Int2LongOpenHashTable() {// for Externalizable
        this._loadFactor = DEFAULT_LOAD_FACTOR;
        this._growFactor = DEFAULT_GROW_FACTOR;
    }

    public Int2LongOpenHashTable(int size) {
        this(size, DEFAULT_LOAD_FACTOR, DEFAULT_GROW_FACTOR, true);
    }

    public Int2LongOpenHashTable(int size, float loadFactor, float growFactor) {
        this(size, loadFactor, growFactor, true);
    }

    protected Int2LongOpenHashTable(int size, float loadFactor, float growFactor, boolean forcePrime) {
        if (size < 1) {
            throw new IllegalArgumentException();
        }
        this._loadFactor = loadFactor;
        this._growFactor = growFactor;
        int actualSize = forcePrime ? Primes.findLeastPrimeNumber(size) : size;
        this._keys = new int[actualSize];
        this._values = new long[actualSize];
        this._states = new byte[actualSize];
        this._used = 0;
        this._threshold = (int) (actualSize * _loadFactor);
    }

    public Int2LongOpenHashTable(@Nonnull int[] keys, @Nonnull long[] values,
            @Nonnull byte[] states, int used) {
        this._loadFactor = DEFAULT_LOAD_FACTOR;
        this._growFactor = DEFAULT_GROW_FACTOR;
        this._keys = keys;
        this._values = values;
        this._states = states;
        this._used = used;
        this._threshold = keys.length;
    }

    @Nonnull
    public static Int2LongOpenHashTable newInstance() {
        return new Int2LongOpenHashTable(DEFAULT_SIZE);
    }

    public void defaultReturnValue(long v) {
        this.defaultReturnValue = v;
    }

    @Nonnull
    public int[] getKeys() {
        return _keys;
    }

    @Nonnull
    public long[] getValues() {
        return _values;
    }

    @Nonnull
    public byte[] getStates() {
        return _states;
    }

    public boolean containsKey(final int key) {
        return findKey(key) >= 0;
    }

    /**
     * @return -1.f if not found
     */
    public long get(final int key) {
        final int i = findKey(key);
        if (i < 0) {
            return defaultReturnValue;
        }
        return _values[i];
    }

    public long put(final int key, final long value) {
        final int hash = keyHash(key);
        int keyLength = _keys.length;
        int keyIdx = hash % keyLength;

        boolean expanded = preAddEntry(keyIdx);
        if (expanded) {
            keyLength = _keys.length;
            keyIdx = hash % keyLength;
        }

        final int[] keys = _keys;
        final long[] values = _values;
        final byte[] states = _states;

        if (states[keyIdx] == FULL) {// double hashing
            if (keys[keyIdx] == key) {
                long old = values[keyIdx];
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
                    long old = values[keyIdx];
                    values[keyIdx] = value;
                    return old;
                }
            }
        }
        keys[keyIdx] = key;
        values[keyIdx] = value;
        states[keyIdx] = FULL;
        ++_used;
        return defaultReturnValue;
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

    protected int findKey(final int key) {
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

    public long remove(final int key) {
        final int[] keys = _keys;
        final long[] values = _values;
        final byte[] states = _states;
        final int keyLength = keys.length;

        final int hash = keyHash(key);
        int keyIdx = hash % keyLength;
        if (states[keyIdx] != FREE) {
            if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                long old = values[keyIdx];
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
                    return defaultReturnValue;
                }
                if (states[keyIdx] == FULL && keys[keyIdx] == key) {
                    long old = values[keyIdx];
                    states[keyIdx] = REMOVED;
                    --_used;
                    return old;
                }
            }
        }
        return defaultReturnValue;
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

    @Nonnull
    public MapIterator entries() {
        return new MapIterator();
    }

    @Override
    public String toString() {
        int len = size() * 10 + 2;
        final StringBuilder buf = new StringBuilder(len);
        buf.append('{');
        final MapIterator itor = entries();
        while (itor.next() != -1) {
            buf.append(itor.getKey());
            buf.append('=');
            buf.append(itor.getValue());
            if (itor.hasNext()) {
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
        int oldCapacity = _keys.length;
        if (newCapacity <= oldCapacity) {
            throw new IllegalArgumentException("new: " + newCapacity + ", old: " + oldCapacity);
        }
        final int[] newkeys = new int[newCapacity];
        final long[] newValues = new long[newCapacity];
        final byte[] newStates = new byte[newCapacity];
        int used = 0;
        for (int i = 0; i < oldCapacity; i++) {
            if (_states[i] == FULL) {
                used++;
                final int k = _keys[i];
                final long v = _values[i];
                final int hash = keyHash(k);
                int keyIdx = hash % newCapacity;
                if (newStates[keyIdx] == FULL) {// second hashing
                    final int decr = 1 + (hash % (newCapacity - 2));
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
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(_threshold);
        out.writeInt(_used);

        final int[] keys = _keys;
        final int size = keys.length;
        out.writeInt(size);

        final byte[] states = _states;
        writeStates(states, out);

        final long[] values = _values;
        for (int i = 0; i < size; i++) {
            if (states[i] != FULL) {
                continue;
            }
            ZigZagLEB128Codec.writeSignedInt(keys[i], out);
            ZigZagLEB128Codec.writeSignedLong(values[i], out);
        }
    }

    @Nonnull
    private static void writeStates(@Nonnull final byte[] status, @Nonnull final DataOutput out)
            throws IOException {
        // write empty states's indexes differentially
        final int size = status.length;
        int cardinarity = 0;
        for (int i = 0; i < size; i++) {
            if (status[i] != FULL) {
                cardinarity++;
            }
        }
        out.writeInt(cardinarity);
        if (cardinarity == 0) {
            return;
        }
        int prev = 0;
        for (int i = 0; i < size; i++) {
            if (status[i] != FULL) {
                int diff = i - prev;
                assert (diff >= 0);
                VariableByteCodec.encodeUnsignedInt(diff, out);
                prev = i;
            }
        }
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        this._threshold = in.readInt();
        this._used = in.readInt();

        final int size = in.readInt();
        final int[] keys = new int[size];
        final long[] values = new long[size];
        final byte[] states = new byte[size];
        readStates(in, states);

        for (int i = 0; i < size; i++) {
            if (states[i] != FULL) {
                continue;
            }
            keys[i] = ZigZagLEB128Codec.readSignedInt(in);
            values[i] = ZigZagLEB128Codec.readSignedLong(in);
        }

        this._keys = keys;
        this._values = values;
        this._states = states;
    }

    @Nonnull
    private static void readStates(@Nonnull final DataInput in, @Nonnull final byte[] status)
            throws IOException {
        // read non-empty states differentially
        final int cardinarity = in.readInt();
        Arrays.fill(status, IntOpenHashTable.FULL);
        int prev = 0;
        for (int j = 0; j < cardinarity; j++) {
            int i = VariableByteCodec.decodeUnsignedInt(in) + prev;
            status[i] = IntOpenHashTable.FREE;
            prev = i;
        }
    }

    public final class MapIterator {

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

        /**
         * @return -1 if not found
         */
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

        public long getValue() {
            if (lastEntry == -1) {
                throw new IllegalStateException();
            }
            return _values[lastEntry];
        }
    }
}
