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

import static hivemall.utils.lang.ArrayUtils.argmax;
import static hivemall.utils.lang.ArrayUtils.argmin;
import static hivemall.utils.lang.ArrayUtils.argrank;
import static hivemall.utils.lang.ArrayUtils.argsort;
import static hivemall.utils.lang.ArrayUtils.newInstance;
import static hivemall.utils.lang.ArrayUtils.range;
import static hivemall.utils.lang.ArrayUtils.slice;
import static java.lang.Math.abs;

import java.util.Comparator;

import org.apache.commons.collections.ComparatorUtils;
import org.junit.Assert;
import org.junit.Test;

public class ArrayUtilsTest {

    @Test
    public void testNewInstance() {
        Assert.assertArrayEquals(new Integer[2], newInstance(new Integer[] {1, 2, 3}, 2));
    }

    @Test
    public void testSlice() {
        int[] a = new int[] {5, 2, 0, 1};
        Assert.assertArrayEquals(new int[] {1, 2}, slice(a, new int[] {3, 1}));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testSliceIndexOutOfBound() {
        int[] a = new int[] {5, 2, 0, 1};
        Assert.assertArrayEquals(new int[] {1, 2}, slice(a, new int[] {3, 5}));
    }

    @Test
    public void testArgsortTArray() {
        Double[] a = new Double[] {5d, 2d, 0d, 1d};
        Assert.assertArrayEquals(new int[] {2, 3, 1, 0}, argsort(a));
        Assert.assertArrayEquals(new Double[] {0d, 1d, 2d, 5d}, slice(a, argsort(a)));
        Assert.assertArrayEquals(new int[] {3, 2, 0, 1}, argsort(argsort(a)));
    }

    @Test
    public void testArgsortTArrayComparatorOfQsuperT() {
        Double[] a = new Double[] {5d, -2d, 0d, -1d};
        Comparator<Double> cmp = new Comparator<Double>() {
            @Override
            public int compare(Double l, Double r) {
                return Double.compare(abs(l.doubleValue()), abs(r.doubleValue()));
            }
        };
        Assert.assertArrayEquals(new int[] {2, 3, 1, 0}, argsort(a, cmp));
        Assert.assertArrayEquals(new Double[] {0d, -1d, -2d, 5d}, slice(a, argsort(a, cmp)));
    }

    @Test
    public void testArgrankIntArray() {
        int[] a = new int[] {5, 2, 0, 1};
        Assert.assertArrayEquals(new int[] {3, 2, 0, 1}, argrank(a));
    }

    @Test
    public void testArgrankDoubleArray() {
        double[] a = new double[] {5.1d, 2.1d, 0.1d, 1.1d};
        Assert.assertArrayEquals(new int[] {3, 2, 0, 1}, argrank(a));
    }

    @Test
    public void testArgminDoubleArray() {
        double[] a = new double[] {5d, 2d, 0.1d, 1d};
        Assert.assertEquals(2, argmin(a));
        Assert.assertArrayEquals(new double[] {0.1d}, slice(a, argmin(a)), 1e-8);
    }

    @Test
    public void testArgminTArray() {
        Double[] a = new Double[] {5d, 2d, 0.1d, 1d};
        Assert.assertEquals(2, argmin(a));
        Assert.assertArrayEquals(new Double[] {0.1d}, slice(a, argmin(a)));
    }

    @Test
    public void testArgminTArrayComparatorOfQsuperT() {
        Double[] a = new Double[] {5d, -2d, 0.1d, -1d};
        Comparator<Double> cmp = new Comparator<Double>() {
            @Override
            public int compare(Double l, Double r) {
                return Double.compare(abs(l.doubleValue()), abs(r.doubleValue()));
            }
        };
        Assert.assertEquals(2, argmin(a, cmp));
        Assert.assertArrayEquals(new Double[] {0.1d}, slice(a, argmin(a, cmp)));
    }

    @Test
    public void testArgmaxTArray() {
        Double[] a = new Double[] {5d, 2d, 0.1d, 1d};
        Assert.assertEquals(0, argmax(a));
        Assert.assertArrayEquals(new Double[] {5d}, slice(a, argmax(a)));
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testArgmaxTArrayComparatorOfQsuperT() {
        Double[] a = new Double[] {2d, 5d, 0d, null, 1d};
        Assert.assertEquals(2, argmax(a, ComparatorUtils.nullLowComparator(
            ComparatorUtils.reversedComparator(ComparatorUtils.naturalComparator()))));
        Assert.assertEquals(1,
            argmax(a, ComparatorUtils.nullLowComparator(ComparatorUtils.naturalComparator())));
        Assert.assertEquals(3,
            argmax(a, ComparatorUtils.nullHighComparator(ComparatorUtils.naturalComparator())));
    }

    @Test
    public void testRange() {
        Assert.assertArrayEquals(new int[] {0, 1, 2, 3, 4}, range(5));
        Assert.assertArrayEquals(new int[] {1, 2, 3, 4}, range(1, 5));
        Assert.assertArrayEquals(new int[] {0, -1, -2}, range(-3));
        Assert.assertArrayEquals(new int[] {5, 4, 3, 2}, range(5, 1));
        Assert.assertArrayEquals(new int[] {3, 2, 1, 0, -1}, range(3, -2));
    }

}
