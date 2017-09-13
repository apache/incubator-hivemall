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

import java.util.Iterator;
import java.util.Map.Entry;
import java.util.SortedMap;

import org.junit.Assert;
import org.junit.Test;

public class BoundedSortedMapTest {

    @Test
    public void testNaturalOrderTop3() {
        // natural order = ascending
        SortedMap<Integer, Double> map = new BoundedSortedMap<Integer, Double>(3);
        Assert.assertNull(map.put(1, 1.d));
        Assert.assertEquals(Double.valueOf(1.d), map.put(1, 1.1d));
        Assert.assertNull(map.put(4, 4.d));
        Assert.assertNull(map.put(2, 2.d));
        Assert.assertEquals(Double.valueOf(2.d), map.put(2, 2.2d));
        Assert.assertEquals(Double.valueOf(4.d), map.put(3, 3.d));
        Assert.assertEquals(Double.valueOf(3.d), map.put(3, 3.3d));

        Assert.assertEquals(3, map.size());

        Iterator<Entry<Integer, Double>> itor = map.entrySet().iterator();
        Entry<Integer, Double> e = itor.next();
        Assert.assertEquals(Integer.valueOf(1), e.getKey());
        Assert.assertEquals(Double.valueOf(1.1d), e.getValue());
        e = itor.next();
        Assert.assertEquals(Integer.valueOf(2), e.getKey());
        Assert.assertEquals(Double.valueOf(2.2d), e.getValue());
        e = itor.next();
        Assert.assertEquals(Integer.valueOf(3), e.getKey());
        Assert.assertEquals(Double.valueOf(3.3d), e.getValue());
        Assert.assertFalse(itor.hasNext());
    }

    @Test
    public void testReverseOrderTop3() {
        // reverse order = descending
        SortedMap<Integer, Double> map = new BoundedSortedMap<Integer, Double>(3, true);
        Assert.assertNull(map.put(1, 1.d));
        Assert.assertEquals(Double.valueOf(1.d), map.put(1, 1.1d));
        Assert.assertNull(map.put(4, 4.d));
        Assert.assertNull(map.put(2, 2.d));
        Assert.assertEquals(Double.valueOf(2.d), map.put(2, 2.2d));
        Assert.assertEquals(Double.valueOf(1.1d), map.put(3, 3.d));
        Assert.assertEquals(Double.valueOf(3.d), map.put(3, 3.3d));

        Assert.assertEquals(3, map.size());

        Iterator<Entry<Integer, Double>> itor = map.entrySet().iterator();
        Entry<Integer, Double> e = itor.next();
        Assert.assertEquals(Integer.valueOf(4), e.getKey());
        Assert.assertEquals(Double.valueOf(4.d), e.getValue());
        e = itor.next();
        Assert.assertEquals(Integer.valueOf(3), e.getKey());
        Assert.assertEquals(Double.valueOf(3.3d), e.getValue());
        e = itor.next();
        Assert.assertEquals(Integer.valueOf(2), e.getKey());
        Assert.assertEquals(Double.valueOf(2.2d), e.getValue());
        Assert.assertFalse(itor.hasNext());
    }

}
