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
package hivemall.utils.collections.maps;

import hivemall.utils.collections.maps.Long2IntOpenHashTable;
import hivemall.utils.lang.ObjectUtils;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

public class Long2IntOpenHashTableTest {

    @Test
    public void testSize() {
        Long2IntOpenHashTable map = new Long2IntOpenHashTable(16384);
        map.put(1L, 3);
        Assert.assertEquals(3, map.get(1L));
        map.put(1L, 5);
        Assert.assertEquals(5, map.get(1L));
        Assert.assertEquals(1, map.size());
    }

    @Test
    public void testDefaultReturnValue() {
        Long2IntOpenHashTable map = new Long2IntOpenHashTable(16384);
        Assert.assertEquals(0, map.size());
        Assert.assertEquals(-1, map.get(1L));
        int ret = Integer.MAX_VALUE;
        map.defaultReturnValue(ret);
        Assert.assertEquals(ret, map.get(1L));
    }

    @Test
    public void testPutAndGet() {
        Long2IntOpenHashTable map = new Long2IntOpenHashTable(16384);
        final int numEntries = 1000000;
        for (int i = 0; i < numEntries; i++) {
            Assert.assertEquals(-1L, map.put(i, i));
        }
        Assert.assertEquals(numEntries, map.size());
        for (int i = 0; i < numEntries; i++) {
            Assert.assertEquals(i, map.get(i));
        }

        map.clear();
        int i = 0;
        for (long j = 1L + Integer.MAX_VALUE; i < 10000; j += 99L, i++) {
            map.put(j, i);
        }
        Assert.assertEquals(i, map.size());
        i = 0;
        for (long j = 1L + Integer.MAX_VALUE; i < 10000; j += 99L, i++) {
            Assert.assertEquals(i, map.get(j));
        }
    }

    @Test
    public void testSerde() throws IOException, ClassNotFoundException {
        Long2IntOpenHashTable map = new Long2IntOpenHashTable(16384);
        final int numEntries = 1000000;
        for (int i = 0; i < numEntries; i++) {
            Assert.assertEquals(-1, map.put(i, i));
        }

        byte[] b = ObjectUtils.toCompressedBytes(map);
        map = new Long2IntOpenHashTable(16384);
        ObjectUtils.readCompressedObject(b, map);

        Assert.assertEquals(numEntries, map.size());
        for (int i = 0; i < numEntries; i++) {
            Assert.assertEquals(i, map.get(i));
        }
    }

    @Test
    public void testIterator() {
        Long2IntOpenHashTable map = new Long2IntOpenHashTable(1000);
        Long2IntOpenHashTable.IMapIterator itor = map.entries();
        Assert.assertFalse(itor.hasNext());

        final int numEntries = 1000000;
        for (int i = 0; i < numEntries; i++) {
            Assert.assertEquals(-1, map.put(i, i));
        }
        Assert.assertEquals(numEntries, map.size());

        itor = map.entries();
        Assert.assertTrue(itor.hasNext());
        while (itor.hasNext()) {
            Assert.assertFalse(itor.next() == -1);
            long k = itor.getKey();
            int v = itor.getValue();
            Assert.assertEquals(k, v);
        }
        Assert.assertEquals(-1, itor.next());
    }
}
