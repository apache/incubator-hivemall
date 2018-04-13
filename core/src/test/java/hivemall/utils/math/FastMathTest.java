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
package hivemall.utils.math;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class FastMathTest {
    private static final boolean DEBUG = false;

    @SuppressWarnings("deprecation")
    @Test
    public void testFastInverseSquareRootFloat() {
        final Random rnd = new Random(43L);
        for (int i = 0; i < 100; i++) {
            float v = rnd.nextFloat() * (rnd.nextInt(10000) + 1);
            Assert.assertEquals(Math.sqrt(v), FastMath.sqrt(v), 1E-5d);
        }
    }

    @SuppressWarnings("deprecation")
    @Test
    public void testFastInverseSquareRootDouble() {
        final Random rnd = new Random(43L);
        for (int i = 0; i < 100; i++) {
            double v = rnd.nextDouble() * (rnd.nextInt(10000) + 1);
            Assert.assertEquals(Math.sqrt(v), FastMath.sqrt(v), 1E-10d);
        }
    }

    @Test
    public void testSigmoid() {
        final Random rnd = new Random(43L);
        for (int i = 0; i < 100; i++) {
            double v = rnd.nextGaussian() * (rnd.nextInt(10000) + 1);
            Assert.assertEquals(Double.toString(v), MathUtils.sigmoid(v), FastMath.sigmoid(v),
                1E-8d);
        }
    }

    @SuppressWarnings("deprecation")
    @Test
    public void testSqrtPerformance() {
        double result1 = 0d;
        // warm up for Math.sqrt
        for (double x = 1d; x < 4_000_000d; x += 0.25d) {
            result1 += Math.sqrt(x);
        }
        long startTime = System.nanoTime();
        for (double x = 1d; x < 4_000_000d; x += 0.25d) {
            result1 += Math.sqrt(x);
        }
        long elapsedTimeForSqrt = System.nanoTime() - startTime;

        // warm up for FastMath.sqrt
        double result2 = 0d;
        for (double x = 1d; x < 4_000_000d; x += 0.25d) {
            result2 += FastMath.sqrt(x);
        }
        startTime = System.nanoTime();
        for (double x = 1d; x < 4_000_000d; x += 0.25D) {
            result2 += FastMath.sqrt(x);
        }
        long elapsedTimeForFastSqrt = System.nanoTime() - startTime;

        if (DEBUG) {
            System.out.println("elapsedTimeForFastSqrt=" + elapsedTimeForFastSqrt
                    + " and elapsedTimeForSqrt=" + elapsedTimeForSqrt);
        }

        Assert.assertFalse(result1 == 0d);
        Assert.assertFalse(result2 == 0d);
        Assert.assertEquals(result1, result2, 1E-5d);

        /*
        Assert.assertTrue(
            "Expected elapsedTimeForFastSqrt < elapsedTimeForSqrt while elapsedTimeForFastSqrt="
                    + elapsedTimeForFastSqrt + " and elapsedTimeForSqrt=" + elapsedTimeForSqrt,
            elapsedTimeForFastSqrt < elapsedTimeForSqrt);
        */
    }

    public static void main(String[] args) {
        FastMathTest test = new FastMathTest();
        for (int i = 1; i <= 10; i++) {
            System.out.println("-- " + i);
            test.testSqrtPerformance();
        }
    }

}
