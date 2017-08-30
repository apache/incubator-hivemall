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
package hivemall.fm;

import hivemall.fm.Entry.AdaGradEntry;
import hivemall.fm.Entry.FTRLEntry;
import hivemall.fm.FMHyperParameters.FFMHyperParameters;
import hivemall.utils.buffer.HeapBuffer;
import hivemall.utils.collections.lists.LongArrayList;
import hivemall.utils.collections.maps.Int2LongOpenHashTable;
import hivemall.utils.collections.maps.Int2LongOpenHashTable.MapIterator;
import hivemall.utils.lang.NumberUtils;

import java.text.NumberFormat;
import java.util.Locale;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.roaringbitmap.RoaringBitmap;

public final class FFMStringFeatureMapModel extends FieldAwareFactorizationMachineModel {
    private static final int DEFAULT_MAPSIZE = 65536;

    // LEARNING PARAMS
    private float _w0;
    @Nonnull
    private final Int2LongOpenHashTable _map;
    @Nonnull
    private final HeapBuffer _buf;

    @Nonnull
    private final LongArrayList _freelistW;
    @Nonnull
    private final LongArrayList _freelistV;

    private boolean _initV;
    @Nonnull
    private RoaringBitmap _removedV;

    // hyperparams
    private final int _numFields;

    private final int _entrySizeW;
    private final int _entrySizeV;

    // statistics
    private long _bytesAllocated, _bytesUsed;
    private int _numAllocatedW, _numReusedW, _numRemovedW;
    private int _numAllocatedV, _numReusedV, _numRemovedV;

    public FFMStringFeatureMapModel(@Nonnull FFMHyperParameters params) {
        super(params);
        this._w0 = 0.f;
        this._map = new Int2LongOpenHashTable(DEFAULT_MAPSIZE);
        this._buf = new HeapBuffer(HeapBuffer.DEFAULT_CHUNK_SIZE);
        this._freelistW = new LongArrayList();
        this._freelistV = new LongArrayList();
        this._initV = true;
        this._removedV = new RoaringBitmap();
        this._numFields = params.numFields;
        this._entrySizeW = entrySize(1, _useFTRL, _useAdaGrad);
        this._entrySizeV = entrySize(_factor, _useFTRL, _useAdaGrad);
    }

    private static int entrySize(@Nonnegative int factors, boolean ftrl, boolean adagrad) {
        if (ftrl) {
            return FTRLEntry.sizeOf(factors);
        } else if (adagrad) {
            return AdaGradEntry.sizeOf(factors);
        } else {
            return Entry.sizeOf(factors);
        }
    }

    void disableInitV() {
        this._initV = false;
    }

    @Override
    public int getSize() {
        return _map.size();
    }

    @Override
    public float getW0() {
        return _w0;
    }

    @Override
    protected void setW0(float nextW0) {
        this._w0 = nextW0;
    }

    @Override
    public float getW(@Nonnull final Feature x) {
        int j = Feature.toIntFeature(x);

        Entry entry = getEntry(j);
        if (entry == null) {
            return 0.f;
        }
        return entry.getW();
    }

    @Override
    protected void setW(@Nonnull final Feature x, final float nextWi) {
        final int j = Feature.toIntFeature(x);

        Entry entry = getEntry(j);
        if (entry == null) {
            entry = newEntry(j, nextWi);
            long ptr = entry.getOffset();
            _map.put(j, ptr);
        } else {
            entry.setW(nextWi);
        }
    }

    /**
     * @return V_x,yField,f
     */
    @Override
    public float getV(@Nonnull final Feature x, @Nonnull final int yField, final int f) {
        final int j = Feature.toIntFeature(x, yField, _numFields);

        Entry entry = getEntry(j);
        if (entry == null) {
            if (_initV == false) {
                return 0.f;
            } else if (_removedV.contains(j)) {
                return 0.f;
            }
            float[] V = initV();
            entry = newEntry(j, V);
            long ptr = entry.getOffset();
            _map.put(j, ptr);
            return V[f];
        }
        return entry.getV(f);
    }

    @Override
    protected void setV(@Nonnull final Feature x, @Nonnull final int yField, final int f,
            final float nextVif) {
        final int j = Feature.toIntFeature(x, yField, _numFields);

        Entry entry = getEntry(j);
        if (entry == null) {
            if (_initV == false) {
                return;
            } else if (_removedV.contains(j)) {
                return;
            }
            float[] V = initV();
            entry = newEntry(j, V);
            long ptr = entry.getOffset();
            _map.put(j, ptr);
        }
        entry.setV(f, nextVif);
    }

    @Override
    protected Entry getEntryW(@Nonnull final Feature x) {
        final int j = Feature.toIntFeature(x);

        Entry entry = getEntry(j);
        if (entry == null) {
            entry = newEntry(j, 0.f);
            long ptr = entry.getOffset();
            _map.put(j, ptr);
        }
        return entry;
    }

    @Override
    protected Entry getEntryV(@Nonnull final Feature x, @Nonnull final int yField) {
        final int j = Feature.toIntFeature(x, yField, _numFields);

        Entry entry = getEntry(j);
        if (entry == null) {
            if (_initV == false) {
                return null;
            } else if (_removedV.contains(j)) {
                return null;
            }
            float[] V = initV();
            entry = newEntry(j, V);
            long ptr = entry.getOffset();
            _map.put(j, ptr);
        }
        return entry;
    }

    @Override
    protected void removeEntry(@Nonnull final Entry entry) {
        final int j = entry.getKey();
        final long ptr = _map.remove(j);
        if (ptr == -1L) {
            return; // should never be happen.
        }
        entry.clear();
        if (Entry.isEntryW(j)) {
            _freelistW.add(ptr);
            this._numRemovedW++;
            this._bytesUsed -= _entrySizeW;
        } else {
            _removedV.add(j);
            _freelistV.add(ptr);
            this._numRemovedV++;
            this._bytesUsed -= _entrySizeV;
        }
    }

    @Nonnull
    protected final Entry newEntry(final int key, final float W) {
        final long ptr;
        if (_freelistW.isEmpty()) {
            ptr = _buf.allocate(_entrySizeW);
            this._numAllocatedW++;
            this._bytesAllocated += _entrySizeW;
            this._bytesUsed += _entrySizeW;
        } else {// reuse removed entry
            ptr = _freelistW.remove();
            this._numReusedW++;
        }
        final Entry entry;
        if (_useFTRL) {
            entry = new FTRLEntry(_buf, key, ptr);
        } else if (_useAdaGrad) {
            entry = new AdaGradEntry(_buf, key, ptr);
        } else {
            entry = new Entry(_buf, key, ptr);
        }

        entry.setW(W);
        return entry;
    }

    @Nonnull
    protected final Entry newEntry(final int key, @Nonnull final float[] V) {
        final long ptr;
        if (_freelistV.isEmpty()) {
            ptr = _buf.allocate(_entrySizeV);
            this._numAllocatedV++;
            this._bytesAllocated += _entrySizeV;
            this._bytesUsed += _entrySizeV;
        } else {// reuse removed entry
            ptr = _freelistV.remove();
            this._numReusedV++;
        }
        final Entry entry;
        if (_useFTRL) {
            entry = new FTRLEntry(_buf, _factor, key, ptr);
        } else if (_useAdaGrad) {
            entry = new AdaGradEntry(_buf, _factor, key, ptr);
        } else {
            entry = new Entry(_buf, _factor, key, ptr);
        }

        entry.setV(V);
        return entry;
    }

    @Nullable
    private Entry getEntry(final int key) {
        final long ptr = _map.get(key);
        if (ptr == -1L) {
            return null;
        }
        return getEntry(key, ptr);
    }

    @Nonnull
    private Entry getEntry(final int key, @Nonnegative final long ptr) {
        if (Entry.isEntryW(key)) {
            if (_useFTRL) {
                return new FTRLEntry(_buf, key, ptr);
            } else if (_useAdaGrad) {
                return new AdaGradEntry(_buf, key, ptr);
            } else {
                return new Entry(_buf, key, ptr);
            }
        } else {
            if (_useFTRL) {
                return new FTRLEntry(_buf, _factor, key, ptr);
            } else if (_useAdaGrad) {
                return new AdaGradEntry(_buf, _factor, key, ptr);
            } else {
                return new Entry(_buf, _factor, key, ptr);
            }
        }
    }

    @Nonnull
    String getStatistics() {
        final NumberFormat fmt = NumberFormat.getIntegerInstance(Locale.US);
        return "FFMStringFeatureMapModel [bytesAllocated="
                + NumberUtils.prettySize(_bytesAllocated) + ", bytesUsed="
                + NumberUtils.prettySize(_bytesUsed) + ", numAllocatedW="
                + fmt.format(_numAllocatedW) + ", numReusedW=" + fmt.format(_numReusedW)
                + ", numRemovedW=" + fmt.format(_numRemovedW) + ", numAllocatedV="
                + fmt.format(_numAllocatedV) + ", numReusedV=" + fmt.format(_numReusedV)
                + ", numRemovedV=" + fmt.format(_numRemovedV) + "]";
    }

    @Override
    public String toString() {
        return getStatistics();
    }

    @Nonnull
    EntryIterator entries() {
        return new EntryIterator(this);
    }

    static final class EntryIterator {

        @Nonnull
        private final MapIterator dictItor;
        @Nonnull
        private final Entry entryProbeW;
        @Nonnull
        private final Entry entryProbeV;

        EntryIterator(@Nonnull FFMStringFeatureMapModel model) {
            this.dictItor = model._map.entries();
            this.entryProbeW = new Entry(model._buf, 1);
            this.entryProbeV = new Entry(model._buf, model._factor);
        }

        @Nonnull
        Entry getEntryProbeW() {
            return entryProbeW;
        }

        @Nonnull
        Entry getEntryProbeV() {
            return entryProbeV;
        }

        boolean hasNext() {
            return dictItor.hasNext();
        }

        boolean next() {
            return dictItor.next() != -1;
        }

        int getEntryIndex() {
            return dictItor.getKey();
        }

        @Nonnull
        void getEntry(@Nonnull final Entry probe) {
            long offset = dictItor.getValue();
            probe.setOffset(offset);
        }

    }

}
