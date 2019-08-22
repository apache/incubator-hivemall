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
}
