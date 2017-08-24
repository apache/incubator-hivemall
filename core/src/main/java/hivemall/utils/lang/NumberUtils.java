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

import java.text.DecimalFormat;

public final class NumberUtils {

    private NumberUtils() {}

    public static int parseInt(String s) {
        int endIndex = s.length() - 1;
        char last = s.charAt(endIndex);
        if (Character.isLetter(last)) {
            String numstr = s.substring(0, endIndex);
            int i = Integer.parseInt(numstr);
            switch (last) {
                case 'k':
                case 'K':
                    i *= 1000;
                    break;
                case 'm':
                case 'M':
                    i *= 1000000;
                    break;
                case 'g':
                case 'G':
                    i *= 1000000000;
                    break;
                default:
                    throw new NumberFormatException("Invalid number format: " + s);
            }
            return i;
        } else {
            return Integer.parseInt(s);
        }
    }

    public static int parseInt(String s, int defaultValue) {
        if (s == null) {
            return defaultValue;
        }
        return parseInt(s);
    }

    public static String formatNumber(final long number) {
        DecimalFormat f = new DecimalFormat("#,###");
        return f.format(number);
    }

    public static String prettySize(long size) {
        if (size < 0) {
            return "N/A";
        } else {
            if (size < 1024) {
                return size + " bytes";
            } else {
                float kb = size / 1024f;
                if (kb < 1024f) {
                    return String.format("%.1f KiB", kb);
                } else {
                    float mb = kb / 1024f;
                    if (mb < 1024f) {
                        return String.format("%.1f MiB", mb);
                    } else {
                        float gb = mb / 1024f;
                        return String.format("%.2f GiB", gb);
                    }
                }
            }
        }
    }

    public static boolean isFinite(final double v) {
        return (v > Double.NEGATIVE_INFINITY) & (v < Double.POSITIVE_INFINITY);
    }

    public static boolean isFinite(final float v) {
        return (v > Float.NEGATIVE_INFINITY) & (v < Float.POSITIVE_INFINITY);
    }

    public static boolean isDigits(String str) {
        if (str == null || str.length() == 0) {
            return false;
        }
        for (int i = 0, len = str.length(); i < len; i++) {
            if (!Character.isDigit(str.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * @throws ArithmeticException
     */
    public static int castToInt(final long value) {
        final int result = (int) value;
        if (result != value) {
            throw new ArithmeticException("Out of range: " + value);
        }
        return result;
    }

    /**
     * @throws ArithmeticException
     */
    public static short castToShort(final int value) {
        final short result = (short) value;
        if (result != value) {
            throw new ArithmeticException("Out of range: " + value);
        }
        return result;
    }

    /**
     * Cast Double to Float.
     * 
     * @throws ArithmeticException
     */
    public static float castToFloat(final double v) {
        if ((v < Float.MIN_VALUE) || (v > Float.MAX_VALUE)) {
            throw new ArithmeticException("Double value is out of Float range: " + v);
        }
        return (float) v;
    }

    /**
     * Cast Double to Float.
     * 
     * @return v if v is Float range; Float.MIN_VALUE or Float.MAX_VALUE otherwise
     */
    public static float safeCast(final double v) {
        if (v < Float.MIN_VALUE) {
            return Float.MIN_VALUE;
        } else if (v > Float.MAX_VALUE) {
            return Float.MAX_VALUE;
        }
        return (float) v;
    }

    /**
     * Cast Double to Float.
     * 
     * @return v if v is Float range; defaultValue otherwise
     */
    public static float safeCast(final double v, final float defaultValue) {
        if ((v < Float.MIN_VALUE) || (v > Float.MAX_VALUE)) {
            return defaultValue;
        }
        return (float) v;
    }

    public static int toUnsignedShort(final short v) {
        return v & 0xFFFF; // convert to range 0-65535 from -32768-32767.
    }

    public static int toUnsignedInt(final byte x) {
        return ((int) x) & 0xff;
    }

}
