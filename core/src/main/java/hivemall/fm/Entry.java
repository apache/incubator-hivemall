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

import hivemall.utils.buffer.HeapBuffer;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.SizeOf;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

class Entry {

    @Nonnull
    protected final HeapBuffer _buf;
    @Nonnegative
    protected final int _size;
    @Nonnegative
    protected final int _factors;

    // temporary variables used only in training phase
    protected int _key;
    @Nonnegative
    protected long _offset;

    Entry(@Nonnull HeapBuffer buf, int factors) {
        this._buf = buf;
        this._size = Entry.sizeOf(factors);
        this._factors = factors;
    }

    Entry(@Nonnull HeapBuffer buf, int key, @Nonnegative long offset) {
        this(buf, 1, key, offset);
    }

    Entry(@Nonnull HeapBuffer buf, int factors, int key, @Nonnegative long offset) {
        this(buf, factors, Entry.sizeOf(factors), key, offset);
    }

    private Entry(@Nonnull HeapBuffer buf, int factors, int size, int key, @Nonnegative long offset) {
        this._buf = buf;
        this._size = size;
        this._factors = factors;
        this._key = key;
        this._offset = offset;
    }

    final int getSize() {
        return _size;
    }

    final int getKey() {
        return _key;
    }

    final long getOffset() {
        return _offset;
    }

    final void setOffset(final long offset) {
        this._offset = offset;
    }

    final float getW() {
        return _buf.getFloat(_offset);
    }

    final void setW(final float value) {
        _buf.putFloat(_offset, value);
    }

    final void getV(@Nonnull final float[] Vf) {
        final long offset = _offset;
        final int len = Vf.length;
        for (int f = 0; f < len; f++) {
            long index = offset + SizeOf.FLOAT * f;
            Vf[f] = _buf.getFloat(index);
        }
    }

    final void setV(@Nonnull final float[] Vf) {
        final long offset = _offset;
        final int len = Vf.length;
        for (int f = 0; f < len; f++) {
            long index = offset + SizeOf.FLOAT * f;
            _buf.putFloat(index, Vf[f]);
        }
    }

    final float getV(final int f) {
        long index = _offset + SizeOf.FLOAT * f;
        return _buf.getFloat(index);
    }

    final void setV(final int f, final float value) {
        long index = _offset + SizeOf.FLOAT * f;
        _buf.putFloat(index, value);
    }

    double getSumOfSquaredGradients(@Nonnegative int f) {
        throw new UnsupportedOperationException();
    }

    void addGradient(@Nonnegative int f, float grad) {
        throw new UnsupportedOperationException();
    }

    final float updateZ(final float gradW, final float alpha) {
        float w = getW();
        return updateZ(0, w, gradW, alpha);
    }

    float updateZ(@Nonnegative int f, float W, float gradW, float alpha) {
        throw new UnsupportedOperationException();
    }

    final double updateN(final float gradW) {
        return updateN(0, gradW);
    }

    double updateN(@Nonnegative int f, float gradW) {
        throw new UnsupportedOperationException();
    }

    boolean removable() {
        if (!isEntryW(_key)) {// entry for V
            final long offset = _offset;
            for (int f = 0; f < _factors; f++) {
                final float Vf = _buf.getFloat(offset + SizeOf.FLOAT * f);
                if (!MathUtils.closeToZero(Vf, 1E-9f)) {
                    return false;
                }
            }
        }
        return true;
    }

    void clear() {};

    static int sizeOf(@Nonnegative final int factors) {
        Preconditions.checkArgument(factors >= 1, "Factors must be greather than 0: " + factors);
        return SizeOf.FLOAT * factors;
    }

    static boolean isEntryW(final int i) {
        return i < 0;
    }

    @Override
    public String toString() {
        if (Entry.isEntryW(_key)) {
            return "W=" + getW();
        } else {
            float[] Vf = new float[_factors];
            getV(Vf);
            return "V=" + Arrays.toString(Vf);
        }
    }

    static final class AdaGradEntry extends Entry {

        final long _gg_offset;

        AdaGradEntry(@Nonnull HeapBuffer buf, int key, @Nonnegative long offset) {
            this(buf, 1, key, offset);
        }

        AdaGradEntry(@Nonnull HeapBuffer buf, @Nonnegative int factors, int key,
                @Nonnegative long offset) {
            super(buf, factors, AdaGradEntry.sizeOf(factors), key, offset);
            this._gg_offset = _offset + Entry.sizeOf(factors);
        }

        @Override
        double getSumOfSquaredGradients(@Nonnegative final int f) {
            Preconditions.checkArgument(f >= 0);

            long offset = _gg_offset + SizeOf.DOUBLE * f;
            return _buf.getDouble(offset);
        }

        @Override
        void addGradient(@Nonnegative final int f, final float grad) {
            Preconditions.checkArgument(f >= 0);

            long offset = _gg_offset + SizeOf.DOUBLE * f;
            double v = _buf.getDouble(offset);
            v += grad * grad;
            _buf.putDouble(offset, v);
        }

        @Override
        void clear() {
            for (int f = 0; f < _factors; f++) {
                long offset = _gg_offset + SizeOf.DOUBLE * f;
                _buf.putDouble(offset, 0.d);
            }
        }

        static int sizeOf(@Nonnegative final int factors) {
            return Entry.sizeOf(factors) + SizeOf.DOUBLE * factors;
        }

        @Override
        public String toString() {
            final double[] gg = new double[_factors];
            for (int f = 0; f < _factors; f++) {
                gg[f] = getSumOfSquaredGradients(f);
            }
            return super.toString() + ", gg=" + Arrays.toString(gg);
        }

    }

    static final class FTRLEntry extends Entry {

        final long _z_offset;

        FTRLEntry(@Nonnull HeapBuffer buf, int key, long offset) {
            this(buf, 1, key, offset);
        }

        FTRLEntry(@Nonnull HeapBuffer buf, @Nonnegative int factors, int key, long offset) {
            super(buf, factors, FTRLEntry.sizeOf(factors), key, offset);
            this._z_offset = _offset + Entry.sizeOf(factors);
        }

        @Override
        float updateZ(final int f, final float W, final float gradW, final float alpha) {
            Preconditions.checkArgument(f >= 0);

            final long zOffset = offsetZ(f);

            final float z = _buf.getFloat(zOffset);
            final double n = _buf.getFloat(offsetN(f)); // implicit cast to float

            double gg = gradW * gradW;
            float sigma = (float) ((Math.sqrt(n + gg) - Math.sqrt(n)) / alpha);

            final float newZ = z + gradW - sigma * W;
            if (!NumberUtils.isFinite(newZ)) {
                throw new IllegalStateException("Got newZ " + newZ + " where z=" + z + ", gradW="
                        + gradW + ", sigma=" + sigma + ", W=" + W + ", n=" + n + ", gg=" + gg
                        + ", alpha=" + alpha);
            }
            _buf.putFloat(zOffset, newZ);
            return newZ;
        }

        @Override
        double updateN(final int f, final float gradW) {
            Preconditions.checkArgument(f >= 0);

            final long nOffset = offsetN(f);

            final double n = _buf.getFloat(nOffset);
            final double newN = n + gradW * gradW;
            if (!NumberUtils.isFinite(newN)) {
                throw new IllegalStateException("Got newN " + newN + " where n=" + n + ", gradW="
                        + gradW);
            }
            _buf.putFloat(nOffset, NumberUtils.castToFloat(newN)); // cast may throw ArithmeticException
            return newN;
        }

        private long offsetZ(@Nonnegative final int f) {
            return _z_offset + SizeOf.FLOAT * f;
        }

        private long offsetN(@Nonnegative final int f) {
            return _z_offset + SizeOf.FLOAT * (_factors + f);
        }

        @Override
        void clear() {
            for (int f = 0; f < _factors; f++) {
                _buf.putFloat(offsetZ(f), 0.f);
                _buf.putFloat(offsetN(f), 0.f);
            }
        }

        static int sizeOf(@Nonnegative final int factors) {
            return Entry.sizeOf(factors) + (SizeOf.FLOAT + SizeOf.FLOAT) * factors;
        }

        @Override
        public String toString() {
            final float[] Z = new float[_factors];
            final float[] N = new float[_factors];
            for (int f = 0; f < _factors; f++) {
                Z[f] = _buf.getFloat(offsetZ(f));
                N[f] = _buf.getFloat(offsetN(f));
            }
            return super.toString() + ", Z=" + Arrays.toString(Z) + ", N=" + Arrays.toString(N);
        }
    }

}
