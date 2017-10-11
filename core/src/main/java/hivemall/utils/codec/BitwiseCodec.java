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
package hivemall.utils.codec;

import hivemall.utils.io.BitInputStream;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.BitUtils;
import hivemall.utils.lang.StringUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * @link http://en.wikipedia.org/wiki/Elias_gamma_coding
 * @link http://en.wikipedia.org/wiki/Elias_delta_coding
 * @link http://en.wikipedia.org/wiki/Golomb_coding
 * @link http://urchin.earth.li/~twic/Golomb-Rice_Coding.html
 * @link http://www.cs.otago.ac.nz/cosc463/2005/compress.htm
 */
public final class BitwiseCodec implements Cloneable {

    /** optiomal for range 1-1000 */
    public static final int LOG2M4 = 2;

    private byte[] codes;
    private int pos = 0;
    private int pendingBits = 0;
    private int curByte = 0;

    public BitwiseCodec(int bytes) {
        codes = new byte[bytes];
    }

    private BitwiseCodec(BitwiseCodec toClone) {
        byte[] origCodes = toClone.codes;
        this.codes = ArrayUtils.copyOf(origCodes, origCodes.length);
        this.pos = toClone.pos;
        this.pendingBits = toClone.pendingBits;
        this.curByte = toClone.curByte;
    }

    public void reset() {
        pos = 0;
        pendingBits = curByte = 0;
        Arrays.fill(codes, (byte) 0);
    }

    public int putGamma(int val) {
        if (val == 1) {
            putBit(1);
            return 1;
        } else {
            int logbits = 2;
            for (; logbits < 32; logbits++) {
                if (val < (1 << logbits)) {
                    break;
                }
            }
            final int ret = (logbits << 1) - 1;
            if (logbits < 16) {//optimization
                putBits(val, ret);
            } else {
                putBits(0, logbits - 1);
                putBits(val, logbits);
            }
            return ret;
        }
    }

    public int putDelta(int val) {
        assert (val > 0) : val;
        if (val == 1) {
            putBit(1);
            return 1;
        } else {
            int logbits = 2;
            for (; logbits < 32; logbits++) {
                if (val < (1 << logbits)) {
                    break;
                }
            }
            int ret = putGamma(logbits);
            int databits = logbits - 1;
            putBits(val, databits);
            ret += databits;
            return ret;
        }
    }

    public int putGolomb4(final int val) {
        return putGolomb(val, 4, LOG2M4);
    }

    public int putGolomb(final int val, final int divm) {
        return putGolomb(val, divm, BitUtils.mostSignificantBit(divm));
    }

    public int putGolomb(final int val, final int m, final int logm) {
        int quotient = val / m;
        for (int i = 0; i < quotient; i++) {
            putBit(0);
        }
        putBit(1);
        int remainder = val % m;
        int mask = 1 << (logm - 1);
        for (int i = logm; i >= 0; i--) {
            putBit(remainder & mask);
            mask >>>= 1;
        }
        int bits = quotient + 1 + logm;
        return bits;
    }

    public long putGolombL(final long val, final long m, final int logm) {
        long quotient = val / m;
        if (quotient > 0x7fffffffL) {
            for (long i = 0; i < quotient; i++) {
                putBit(0);
            }
        } else if (quotient > 0) {
            final int qi = (int) quotient;
            for (int i = 0; i < qi; i++) {
                putBit(0);
            }
        }
        putBit(1);
        long remainder = val % m;
        long mask = 1 << (logm - 1);
        for (int i = logm; i >= 0; i--) {
            final long b = remainder & mask;
            putBit(b == 0L ? 0 : 1);
            mask >>>= 1;
        }
        long bits = quotient + 1 + logm;
        return bits;
    }

    public void putBits(int val, int num) {
        for (int i = num - 1; i >= 0; i--) {
            putBit((val >> i) & 0x01);
        }
    }

    public void putBit(int b) {
        curByte = (curByte << 1) | b;
        pendingBits++;

        if (pendingBits == 8) {
            if (pos >= codes.length) {
                expand();
            }
            codes[pos++] = (byte) (curByte & 0xff);
            curByte = 0;
            pendingBits = 0;
        }
    }

    public void allocateNewByte(byte b) {
        padVector();
        if (pos >= codes.length) {
            expand();
        }
        codes[pos++] = b;
        curByte = 0;
        pendingBits = 0;
    }

    private void expand() {
        byte[] oldVector = codes;
        int oldSize = oldVector.length;
        int newSize = (int) (oldSize * 1.75);
        byte[] newVector = new byte[newSize];
        System.arraycopy(oldVector, 0, newVector, 0, oldSize);
        this.codes = newVector;
    }

    private void padVector() {
        if (pendingBits == 0) {
            return;
        }
        if (pos >= codes.length) {
            expand();
        }
        codes[pos++] = (byte) ((curByte << (8 - pendingBits)) & 0xff);
    }

    public byte[] getBytes() {
        padVector();
        final byte[] copy = new byte[pos];
        System.arraycopy(codes, 0, copy, 0, pos);
        return copy;
    }

    public void writeTo(final OutputStream out, final boolean clear) throws IOException {
        padVector();
        IOUtils.writeInt(pos, out);
        out.write(codes, 0, pos);
        if (clear) {
            reset();
        }
    }

    public void getBytes_clear(final OutputStream out) throws IOException {
        padVector();
        out.write(codes, 0, pos);
        reset();
    }

    @Override
    public String toString() {
        return StringUtils.toBitString(getBytes());
    }

    @Override
    public BitwiseCodec clone() {
        BitwiseCodec cloned = new BitwiseCodec(this);
        return cloned;
    }

    public static long requiredBytesGolomb(final long val, final long divm, final int logm) {
        long quotient = val / divm;
        long bits = quotient + 1 + logm;
        return bits;
    }

    public static long decodeGolombL(final byte[] b, final long m, final int logm) {
        long unary = 0;
        final int blen = b.length;
        for (int i = 0; i < blen; i++) {
            int msb = BitUtils.mostSignificantBit(b[i]);
            if (msb == 7) { // 1000
                break;
            } else if (msb == -1) {// 0000
                unary += 8;
            } else {
                unary += (7 - msb);
                break;
            }
        }
        /* Get the first q bits (we may need (q+1) actually) */
        long remainder = BitUtils.getLongBits(b, unary + 1, logm);

        long ret = unary * m + remainder;
        return ret;
    }

    public static long decodeGolombL(final InputStream in, final long m, final int logm)
            throws IOException {
        final BitInputStream bis = new BitInputStream(in);

        long unary = 0;
        while (!bis.readBit()) {// loop until '1' is found
            unary++;
        }

        /* Get the first q bits (we may need (q+1) actually) */
        long remainder = BitUtils.getLongBits(bis, logm);

        long ret = unary * m + remainder;
        return ret;
    }

}
