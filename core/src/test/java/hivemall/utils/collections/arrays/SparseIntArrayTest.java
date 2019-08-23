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
package hivemall.utils.collections.arrays;

import hivemall.utils.function.Consumer;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class SparseIntArrayTest {

    @Test
    public void testDense() {
        int size = 1000;
        Random rand = new Random(31);
        int[] expected = new int[size];
        IntArray actual = new SparseIntArray(10);
        for (int i = 0; i < size; i++) {
            int r = rand.nextInt(size);
            expected[i] = r;
            actual.put(i, r);
        }
        for (int i = 0; i < size; i++) {
            Assert.assertEquals(expected[i], actual.get(i));
        }
    }

    @Test
    public void testSparse() {
        int size = 1000;
        Random rand = new Random(31);
        int[] expected = new int[size];
        SparseIntArray actual = new SparseIntArray(10);
        for (int i = 0; i < size; i++) {
            int key = rand.nextInt(size);
            int v = rand.nextInt();
            expected[key] = v;
            actual.put(key, v);
        }
        for (int i = 0; i < actual.size(); i++) {
            int key = actual.keyAt(i);
            Assert.assertEquals(expected[key], actual.get(key, 0));
        }
    }

    @Test
    public void testAppend() {
        int[] a1 = new int[500];
        for (int i = 0; i < a1.length; i++) {
            a1[i] = i;
        }
        SparseIntArray array = new SparseIntArray(a1);
        for (int i = 0; i < a1.length; i++) {
            Assert.assertEquals(a1[i], array.get(i));
        }
        int[] a2 = new int[100];
        for (int i = 0; i < 100; i++) {
            a2[i] = a1[a1.length - 1] + i;
        }
        array.append(a1.length - 9, a2);
        Assert.assertEquals(a1.length + a2.length - 9, array.size());
    }

    @Test
    public void testAppend2() {
        int[] a1 = new int[500];
        for (int i = 0; i < a1.length; i++) {
            a1[i] = i;
        }
        SparseIntArray array = new SparseIntArray(a1);
        for (int i = 0; i < a1.length; i++) {
            Assert.assertEquals(a1[i], array.get(i));
        }
        int[] a2 = new int[100];
        for (int i = 0; i < 100; i++) {
            a2[i] = a1[a1.length - 1] + i;
        }
        array.append(a1.length - 9, a2, 0, a2.length);
        Assert.assertEquals(a1.length + a2.length - 9, array.size());
    }

    @Test
    public void testConsume() {
        final Random rng = new Random(43L);
        int[] keys = new int[500];
        int[] values = new int[keys.length];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = i * 2;
            values[i] = rng.nextInt(1000);
        }
        final SparseIntArray actual = new SparseIntArray(keys, values, keys.length);
        Assert.assertEquals(500, actual.size());

        actual.consume(10, 30, new Consumer() {
            @Override
            public void accept(int i, int value) {
                actual.put(i, value);
                actual.put(i + 1, value);
            }
        });

        int lastKey = actual.lastKey();
        Assert.assertEquals(998, lastKey);
        actual.append(lastKey, new int[] {-1, -2, -3});

        Assert.assertEquals(512, actual.size());
        Assert.assertEquals(-1, actual.get(998));
        Assert.assertEquals(-2, actual.get(999));
        Assert.assertEquals(-3, actual.get(1000));

        for (int i = 10; i < 30; i += 2) {
            Assert.assertEquals(actual.get(i), actual.get(i + 1));
        }
    }
}
