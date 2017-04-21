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

import hivemall.utils.collections.arrays.SparseDoubleArray;

import org.junit.Assert;
import org.junit.Test;

public class DoubleArrayTest {

    @Test
    public void testSparseDoubleArrayToArray() {
        SparseDoubleArray array = new SparseDoubleArray(3);
        for (int i = 0; i < 10; i++) {
            array.put(i, 10 + i);
        }
        Assert.assertEquals(10, array.size());
        Assert.assertEquals(10, array.toArray(false).length);

        double[] copied = array.toArray(true);
        Assert.assertEquals(10, copied.length);
        for (int i = 0; i < 10; i++) {
            Assert.assertEquals(10 + i, copied[i], 0.d);
        }
    }

    @Test
    public void testSparseDoubleArrayClear() {
        SparseDoubleArray array = new SparseDoubleArray(3);
        for (int i = 0; i < 10; i++) {
            array.put(i, 10 + i);
        }
        array.clear();
        Assert.assertEquals(0, array.size());
        Assert.assertEquals(0, array.get(0), 0.d);
        for (int i = 0; i < 5; i++) {
            array.put(i, 100 + i);
        }
        Assert.assertEquals(5, array.size());
        for (int i = 0; i < 5; i++) {
            Assert.assertEquals(100 + i, array.get(i), 0.d);
        }
    }

}
