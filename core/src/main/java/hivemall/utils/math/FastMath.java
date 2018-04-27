/*
 * Copyright 2012-2015 Jeff Hain
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * =============================================================================
 * Notice of fdlibm package this program is partially derived from:
 *
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * =============================================================================
 */
// This file contains a modified version of Jafama's FastMath:
// https://github.com/jeffhain/jafama/blob/master/src/main/java/net/jafama/FastMath.java
package hivemall.utils.math;

import hivemall.annotations.Experimental;

@Experimental
public final class FastMath {

    private FastMath() {}

    @Deprecated
    public static float sqrt(final float x) {
        return x * invSqrt(x);
    }

    @Deprecated
    public static double sqrt(final double x) {
        return x * invSqrt(x);
    }

    /**
     * https://en.wikipedia.org/wiki/Fast_inverse_square_root
     */
    @Deprecated
    public static float invSqrt(final float x) {
        final float hx = 0.5f * x;
        int i = 0x5f375a86 - (Float.floatToRawIntBits(x) >>> 1);
        float y = Float.intBitsToFloat(i);
        y *= (1.5f - hx * y * y); // pass 1
        y *= (1.5f - hx * y * y); // pass 2
        y *= (1.5f - hx * y * y); // pass 3
        //y *= (1.5f - hx * y * y); // pass 4
        // more pass for more accuracy
        return y;
    }

    /**
     * https://en.wikipedia.org/wiki/Fast_inverse_square_root
     */
    @Deprecated
    public static double invSqrt(final double x) {
        final double hx = 0.5d * x;
        long i = 0x5fe6eb50c7b537a9L - (Double.doubleToRawLongBits(x) >>> 1);
        double y = Double.longBitsToDouble(i);
        y *= (1.5d - hx * y * y); // pass 1
        y *= (1.5d - hx * y * y); // pass 2
        y *= (1.5d - hx * y * y); // pass 3
        y *= (1.5d - hx * y * y); // pass 4
        // more pass for more accuracy
        return y;
    }

    public static double log(final double x) {
        return JafamaMath.log(x);
    }

    /**
     * @return log(1+x)
     */
    public static double log1p(final double x) {
        return JafamaMath.log1p(x);
    }

    /**
     * https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
     * 
     * @return e^x
     */
    public static double exp(final double x) {
        return JafamaMath.exp(x);
    }

    /**
     * @return exp(x)-1
     */
    public static double expm1(final double x) {
        return JafamaMath.expm1(x);
    }

    public static double sigmoid(final double x) {
        return 1 / (1 + exp(-x));
    }

    /**
     * Based on Jafama (https://github.com/jeffhain/jafama/) version 2.2.
     */
    private static final class JafamaMath {

        static final double TWO_POW_52 = twoPow(52);

        /**
         * Double.MIN_NORMAL since Java 6.
         */
        static final double DOUBLE_MIN_NORMAL = Double.longBitsToDouble(0x0010000000000000L); // 2.2250738585072014E-308

        // Not storing float/double mantissa size in constants,
        // for 23 and 52 are shorter to read and more
        // bitwise-explicit than some constant's name.

        static final int MIN_DOUBLE_EXPONENT = -1074;
        static final int MAX_DOUBLE_EXPONENT = 1023;

        static final double LOG_2 = StrictMath.log(2.0);

        //--------------------------------------------------------------------------
        // CONSTANTS AND TABLES FOR EXP AND EXPM1
        //--------------------------------------------------------------------------

        static final double EXP_OVERFLOW_LIMIT = Double.longBitsToDouble(0x40862E42FEFA39EFL); // 7.09782712893383973096e+02
        static final double EXP_UNDERFLOW_LIMIT = Double.longBitsToDouble(0xC0874910D52D3051L); // -7.45133219101941108420e+02
        static final int EXP_LO_DISTANCE_TO_ZERO_POT = 0;
        static final int EXP_LO_DISTANCE_TO_ZERO = (1 << EXP_LO_DISTANCE_TO_ZERO_POT);
        static final int EXP_LO_TAB_SIZE_POT = 11;
        static final int EXP_LO_TAB_SIZE = (1 << EXP_LO_TAB_SIZE_POT) + 1;
        static final int EXP_LO_TAB_MID_INDEX = ((EXP_LO_TAB_SIZE - 1) / 2);
        static final int EXP_LO_INDEXING = EXP_LO_TAB_MID_INDEX / EXP_LO_DISTANCE_TO_ZERO;
        static final int EXP_LO_INDEXING_DIV_SHIFT =
                EXP_LO_TAB_SIZE_POT - 1 - EXP_LO_DISTANCE_TO_ZERO_POT;

        static final class MyTExp {
            static final double[] expHiTab =
                    new double[1 + (int) EXP_OVERFLOW_LIMIT - (int) EXP_UNDERFLOW_LIMIT];
            static final double[] expLoPosTab = new double[EXP_LO_TAB_SIZE];
            static final double[] expLoNegTab = new double[EXP_LO_TAB_SIZE];

            static {
                init();
            }

            private static strictfp void init() {
                for (int i = (int) EXP_UNDERFLOW_LIMIT; i <= (int) EXP_OVERFLOW_LIMIT; i++) {
                    expHiTab[i - (int) EXP_UNDERFLOW_LIMIT] = StrictMath.exp(i);
                }
                for (int i = 0; i < EXP_LO_TAB_SIZE; i++) {
                    // x: in [-EXPM1_DISTANCE_TO_ZERO,EXPM1_DISTANCE_TO_ZERO].
                    double x = -EXP_LO_DISTANCE_TO_ZERO + i / (double) EXP_LO_INDEXING;
                    // exp(x)
                    expLoPosTab[i] = StrictMath.exp(x);
                    // 1-exp(-x), accurately computed
                    expLoNegTab[i] = -StrictMath.expm1(-x);
                }
            }
        }

        //--------------------------------------------------------------------------
        // CONSTANTS AND TABLES FOR LOG AND LOG1P
        //--------------------------------------------------------------------------

        static final int LOG_BITS = 12;
        static final int LOG_TAB_SIZE = (1 << LOG_BITS);

        static final class MyTLog {
            static final double[] logXLogTab = new double[LOG_TAB_SIZE];
            static final double[] logXTab = new double[LOG_TAB_SIZE];
            static final double[] logXInvTab = new double[LOG_TAB_SIZE];

            static {
                init();
            }

            private static strictfp void init() {
                for (int i = 0; i < LOG_TAB_SIZE; i++) {
                    // Exact to use inverse of tab size, since it is a power of two.
                    double x = 1 + i * (1.0 / LOG_TAB_SIZE);
                    logXLogTab[i] = StrictMath.log(x);
                    logXTab[i] = x;
                    logXInvTab[i] = 1 / x;
                }
            }
        }

        /**
         * @param value A double value.
         * @return e^value.
         */
        static double exp(final double value) {
            // exp(x) = exp([x])*exp(y)
            // with [x] the integer part of x, and y = x-[x]
            // ===>
            // We find an approximation of y, called z.
            // ===>
            // exp(x) = exp([x])*(exp(z)*exp(epsilon))
            // with epsilon = y - z
            // ===>
            // We have exp([x]) and exp(z) pre-computed in tables, we "just" have to compute exp(epsilon).
            //
            // We use the same indexing (cast to int) to compute x integer part and the
            // table index corresponding to z, to avoid two int casts.
            // Also, to optimize index multiplication and division, we use powers of two,
            // so that we can do it with bits shifts.

            if (value > EXP_OVERFLOW_LIMIT) {
                return Double.POSITIVE_INFINITY;
            } else if (!(value >= EXP_UNDERFLOW_LIMIT)) {
                return (value != value) ? Double.NaN : 0.0;
            }

            final int indexes = (int) (value * EXP_LO_INDEXING);

            final int valueInt;
            if (indexes >= 0) {
                valueInt = (indexes >> EXP_LO_INDEXING_DIV_SHIFT);
            } else {
                valueInt = -((-indexes) >> EXP_LO_INDEXING_DIV_SHIFT);
            }
            final double hiTerm = MyTExp.expHiTab[valueInt - (int) EXP_UNDERFLOW_LIMIT];

            final int zIndex = indexes - (valueInt << EXP_LO_INDEXING_DIV_SHIFT);
            final double y = (value - valueInt);
            final double z = zIndex * (1.0 / EXP_LO_INDEXING);
            final double eps = y - z;
            final double expZ = MyTExp.expLoPosTab[zIndex + EXP_LO_TAB_MID_INDEX];
            final double expEps =
                    (1 + eps * (1 + eps * (1.0 / 2 + eps * (1.0 / 6 + eps * (1.0 / 24)))));
            final double loTerm = expZ * expEps;

            return hiTerm * loTerm;
        }

        /**
         * Much more accurate than exp(value)-1, for arguments (and results) close to zero.
         *
         * @param value A double value.
         * @return e^value-1.
         */
        static double expm1(final double value) {
            // If value is far from zero, we use exp(value)-1.
            //
            // If value is close to zero, we use the following formula:
            // exp(value)-1
            // = exp(valueApprox)*exp(epsilon)-1
            // = exp(valueApprox)*(exp(epsilon)-exp(-valueApprox))
            // = exp(valueApprox)*(1+epsilon+epsilon^2/2!+...-exp(-valueApprox))
            // = exp(valueApprox)*((1-exp(-valueApprox))+epsilon+epsilon^2/2!+...)
            // exp(valueApprox) and exp(-valueApprox) being stored in tables.

            if (Math.abs(value) < EXP_LO_DISTANCE_TO_ZERO) {
                // Taking int part instead of rounding, which takes too long.
                int i = (int) (value * EXP_LO_INDEXING);
                double delta = value - i * (1.0 / EXP_LO_INDEXING);
                return MyTExp.expLoPosTab[i + EXP_LO_TAB_MID_INDEX] * (MyTExp.expLoNegTab[i
                        + EXP_LO_TAB_MID_INDEX]
                        + delta * (1 + delta * (1.0 / 2
                                + delta * (1.0 / 6 + delta * (1.0 / 24 + delta * (1.0 / 120))))));
            } else {
                return exp(value) - 1;
            }
        }

        /**
         * @param value A double value.
         * @return Value logarithm (base e).
         */
        static double log(double value) {
            if (value > 0.0) {
                if (value == Double.POSITIVE_INFINITY) {
                    return Double.POSITIVE_INFINITY;
                }

                // For normal values not close to 1.0, we use the following formula:
                // log(value)
                // = log(2^exponent*1.mantissa)
                // = log(2^exponent) + log(1.mantissa)
                // = exponent * log(2) + log(1.mantissa)
                // = exponent * log(2) + log(1.mantissaApprox) + log(1.mantissa/1.mantissaApprox)
                // = exponent * log(2) + log(1.mantissaApprox) + log(1+epsilon)
                // = exponent * log(2) + log(1.mantissaApprox) + epsilon-epsilon^2/2+epsilon^3/3-epsilon^4/4+...
                // with:
                // 1.mantissaApprox <= 1.mantissa,
                // log(1.mantissaApprox) in table,
                // epsilon = (1.mantissa/1.mantissaApprox)-1
                //
                // To avoid bad relative error for small results,
                // values close to 1.0 are treated aside, with the formula:
                // log(x) = z*(2+z^2*((2.0/3)+z^2*((2.0/5))+z^2*((2.0/7))+...)))
                // with z=(x-1)/(x+1)

                double h;
                if (value > 0.95) {
                    if (value < 1.14) {
                        double z = (value - 1.0) / (value + 1.0);
                        double z2 = z * z;
                        return z * (2 + z2 * ((2.0 / 3) + z2 * ((2.0 / 5)
                                + z2 * ((2.0 / 7) + z2 * ((2.0 / 9) + z2 * ((2.0 / 11)))))));
                    }
                    h = 0.0;
                } else if (value < DOUBLE_MIN_NORMAL) {
                    // Ensuring value is normal.
                    value *= TWO_POW_52;
                    // log(x*2^52)
                    // = log(x)-ln(2^52)
                    // = log(x)-52*ln(2)
                    h = -52 * LOG_2;
                } else {
                    h = 0.0;
                }

                int valueBitsHi = (int) (Double.doubleToRawLongBits(value) >> 32);
                int valueExp = (valueBitsHi >> 20) - MAX_DOUBLE_EXPONENT;
                // Getting the first LOG_BITS bits of the mantissa.
                int xIndex = ((valueBitsHi << 12) >>> (32 - LOG_BITS));

                // 1.mantissa/1.mantissaApprox - 1
                double z = (value * twoPowNormalOrSubnormal(-valueExp)) * MyTLog.logXInvTab[xIndex]
                        - 1;

                z *= (1 - z * ((1.0 / 2) - z * ((1.0 / 3))));

                return h + valueExp * LOG_2 + (MyTLog.logXLogTab[xIndex] + z);

            } else if (value == 0.0) {
                return Double.NEGATIVE_INFINITY;
            } else { // value < 0.0, or value is NaN
                return Double.NaN;
            }
        }

        /**
         * Much more accurate than log(1+value), for arguments (and results) close to zero.
         *
         * @param value A double value.
         * @return Logarithm (base e) of (1+value).
         */
        static double log1p(final double value) {
            if (value > -1.0) {
                if (value == Double.POSITIVE_INFINITY) {
                    return Double.POSITIVE_INFINITY;
                }

                // ln'(x) = 1/x
                // so
                // log(x+epsilon) ~= log(x) + epsilon/x
                //
                // Let u be 1+value rounded:
                // 1+value = u+epsilon
                //
                // log(1+value)
                // = log(u+epsilon)
                // ~= log(u) + epsilon/value
                // We compute log(u) as done in log(double), and then add the corrective term.

                double valuePlusOne = 1.0 + value;
                if (valuePlusOne == 1.0) {
                    return value;
                } else if (Math.abs(value) < 0.15) {
                    double z = value / (value + 2.0);
                    double z2 = z * z;
                    return z * (2 + z2 * ((2.0 / 3) + z2 * ((2.0 / 5)
                            + z2 * ((2.0 / 7) + z2 * ((2.0 / 9) + z2 * ((2.0 / 11)))))));
                }

                int valuePlusOneBitsHi =
                        (int) (Double.doubleToRawLongBits(valuePlusOne) >> 32) & 0x7FFFFFFF;
                int valuePlusOneExp = (valuePlusOneBitsHi >> 20) - MAX_DOUBLE_EXPONENT;
                // Getting the first LOG_BITS bits of the mantissa.
                int xIndex = ((valuePlusOneBitsHi << 12) >>> (32 - LOG_BITS));

                // 1.mantissa/1.mantissaApprox - 1
                double z = (valuePlusOne * twoPowNormalOrSubnormal(-valuePlusOneExp))
                        * MyTLog.logXInvTab[xIndex] - 1;

                z *= (1 - z * ((1.0 / 2) - z * (1.0 / 3)));

                // Adding epsilon/valuePlusOne to z,
                // with
                // epsilon = value - (valuePlusOne-1)
                // (valuePlusOne + epsilon ~= 1+value (not rounded))

                return valuePlusOneExp * LOG_2 + MyTLog.logXLogTab[xIndex]
                        + (z + (value - (valuePlusOne - 1)) / valuePlusOne);
            } else if (value == -1.0) {
                return Double.NEGATIVE_INFINITY;
            } else { // value < -1.0, or value is NaN
                return Double.NaN;
            }
        }

        /**
         * @param power Must be in normal or subnormal values range.
         */
        private static double twoPowNormalOrSubnormal(final int power) {
            if (power <= -MAX_DOUBLE_EXPONENT) { // Not normal.
                return Double.longBitsToDouble(
                    0x0008000000000000L >> (-(power + MAX_DOUBLE_EXPONENT)));
            } else { // Normal.
                return Double.longBitsToDouble(((long) (power + MAX_DOUBLE_EXPONENT)) << 52);
            }
        }

        /**
         * Returns the exact result, provided it's in double range, i.e. if power is in
         * [-1074,1023].
         *
         * @param power An int power.
         * @return 2^power as a double, or +-Infinity in case of overflow.
         */
        private static double twoPow(final int power) {
            if (power <= -MAX_DOUBLE_EXPONENT) { // Not normal.
                if (power >= MIN_DOUBLE_EXPONENT) { // Subnormal.
                    return Double.longBitsToDouble(
                        0x0008000000000000L >> (-(power + MAX_DOUBLE_EXPONENT)));
                } else { // Underflow.
                    return 0.0;
                }
            } else if (power > MAX_DOUBLE_EXPONENT) { // Overflow.
                return Double.POSITIVE_INFINITY;
            } else { // Normal.
                return Double.longBitsToDouble(((long) (power + MAX_DOUBLE_EXPONENT)) << 52);
            }
        }

    }

}
