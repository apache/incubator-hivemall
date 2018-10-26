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

import javax.annotation.Nonnull;

public final class Primitives {
    public static final int INT_BYTES = Integer.SIZE / Byte.SIZE;
    public static final int DOUBLE_BYTES = Double.SIZE / Byte.SIZE;

    public static final Byte TRUE_BYTE = 1;
    public static final Byte FALSE_BYTE = 0;

    private Primitives() {}

    public static short parseShort(final String s, final short defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Short.parseShort(s);
    }

    public static int parseInt(final String s, final int defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Integer.parseInt(s);
    }

    public static long parseLong(final String s, final long defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Long.parseLong(s);
    }

    public static float parseFloat(final String s, final float defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Float.parseFloat(s);
    }

    public static double parseDouble(final String s, final double defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Double.parseDouble(s);
    }

    public static boolean parseBoolean(final String s, final boolean defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return Boolean.parseBoolean(s);
    }

    public static double doubleValue(final Double v, final double defaultValue) {
        if (v == null) {
            return defaultValue;
        }
        return v.doubleValue();
    }

    public static boolean isFinite(final double value) {
        return Double.NEGATIVE_INFINITY < value && value < Double.POSITIVE_INFINITY;
    }

    public static int compare(final int x, final int y) {
        return (x < y) ? -1 : ((x == y) ? 0 : 1);
    }

    public static void putChar(final byte[] b, final int off, final char val) {
        b[off + 1] = (byte) (val >>> 0);
        b[off] = (byte) (val >>> 8);
    }

    public static long toLong(final int high, final int low) {
        return ((long) high << 32) | ((long) low & 0xffffffffL);
    }

    public static int getHigh(final long key) {
        return (int) (key >>> 32) & 0xffffffff;
    }

    public static int getLow(final long key) {
        return (int) key & 0xffffffff;
    }

    @Nonnull
    public static byte[] toBytes(long l) {
        final byte[] retVal = new byte[8];
        for (int i = 0; i < 8; i++) {
            retVal[i] = (byte) l;
            l >>= 8;
        }
        return retVal;
    }

    public static int hashCode(final long value) {
        return (int) (value ^ (value >>> 32));
    }

}
