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
package hivemall.utils.lang;

import hivemall.utils.io.BitInputStream;

import java.io.IOException;
import java.util.BitSet;

import javax.annotation.Nonnull;

public final class BitUtils {

    private BitUtils() {}

    public static long getLongBits(final byte[] bytes, final long bitOffset, final long bitLen) {
        assert (bitLen <= 64) : bitLen;
        long result = 0;
        final int byteslen = bytes.length;
        final long last = bitOffset + bitLen - 1;
        for (long bi = bitOffset; bi <= last; bi++) {
            final long lquot = bi >>> 3; // b / 8
            assert (lquot <= 0x7fffffffL) : "Illegal quot: " + lquot;
            final int quot = (int) lquot;
            if (quot >= byteslen) {
                throw new ArrayIndexOutOfBoundsException("quot(" + quot + ") >= bytes.length("
                        + byteslen + ')');
            }
            final byte b = bytes[quot];
            final int rem = (int) (bi & 7); // b % 8
            final int mask = 0x80 >> rem;
            if ((b & mask) != 0) {
                result |= 1L << (last - bi);
            }
        }
        return result;
    }

    public static long getLongBits(final BitInputStream in, final long bitLen) throws IOException {
        assert (bitLen <= 64) : bitLen;
        long result = 0;
        final long last = bitLen - 1;
        for (long bi = 0; bi <= last; bi++) {
            final boolean frag = in.readBit();
            if (frag) {
                result |= 1L << (last - bi);
            }
        }
        return result;
    }

    public static BitSet toBitSet(final String s) {
        final int len = s.length();
        final BitSet result = new BitSet(len);
        for (int i = 0; i < len; i++) {
            if (s.charAt(i) == '1') {
                result.set(len - i - 1);
            }
        }
        return result;
    }

    public static String toBinaryString(final BitSet bits) {
        final int len = bits.length();
        final char[] data = new char[len];
        for (int i = 0; i < len; i++) {
            data[len - i - 1] = bits.get(i) ? '1' : '0';
        }
        return String.valueOf(data);
    }

    /**
     * @param nth index starting from 0
     * @return index of n-th set bit or -1 if not found
     */
    public static int indexOfSetBit(@Nonnull final BitSet bits, final int nth) {
        if (nth < 0) {
            throw new IllegalArgumentException("Invalid nth: " + nth);
        }

        int pos = bits.nextSetBit(0);
        for (int i = 0; pos >= 0; pos = bits.nextSetBit(pos + 1), i++) {
            if (i == nth) {
                break;
            }
        }
        return pos;
    }

    public static int indexOfClearBit(@Nonnull final BitSet bits, final int nth, final int lastIndex) {
        int j = bits.nextClearBit(0);
        for (int c = 0; j <= lastIndex; j = bits.nextClearBit(j + 1), c++) {
            if (c == nth) {
                break;
            }
        }
        return j;
    }

    /**
     * Returns the index of the most significant bit in state "true". Returns -1 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 30
     *  0x00000001 --> 0
     *  0x00000000 --> -1
     * </pre>
     */
    public static int mostSignificantBit(final int value) {
        int i = 32;
        while (--i >= 0 && (((1 << i) & value)) == 0);
        return i;
    }

    /**
     * Returns the index of the most significant bit in state "true". Returns -1 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 30
     *  0x00000001 --> 0
     *  0x00000000 --> -1
     * </pre>
     */
    public static int mostSignificantBit(final byte value) {
        int i = 8;
        while (--i >= 0 && (((1 << i) & value)) == 0);
        return i;
    }

    /**
     * Returns the index of the most significant bit in state "true". Returns -1 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 30
     *  0x00000001 --> 0
     *  0x00000000 --> -1
     * </pre>
     */
    public static int mostSignificantBit(final long value) {
        int i = 64;
        while (--i >= 0 && (((1L << i) & value)) == 0);
        return i;
    }

    /**
     * Returns the index of the least significant bit in state "true". Returns 32 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 0
     *  0x00000001 --> 0
     *  0x00000000 --> 32
     * </pre>
     */
    public static int leastSignificantBit(final int value) {
        int i = -1;
        while (++i < 32 && (((1 << i) & value)) == 0);
        return i;
    }

    /**
     * Returns the index of the least significant bit in state "true". Returns 64 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 0
     *  0x00000001 --> 0
     *  0x00000000 --> 32
     * </pre>
     */
    public static int leastSignificantBit(final long value) {
        int i = -1;
        while (++i < 64 && (((1 << i) & value)) == 0);
        return i;
    }

    /**
     * Returns the index of the least significant bit in state "true". Returns 8 if no bit is in
     * state "true".
     * 
     * <pre>
     * Examples:
     *  0x80000000 --> 31
     *  0x7fffffff --> 0
     *  0x00000001 --> 0
     *  0x00000000 --> 32
     * </pre>
     */
    public static int leastSignificantBit(final byte value) {
        int i = -1;
        while (++i < 8 && (((1 << i) & value)) == 0);
        return i;
    }

}
