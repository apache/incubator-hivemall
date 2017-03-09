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
package hivemall.utils.collections.lists;


import org.junit.Assert;
import org.junit.Test;

public class LongArrayListTest {

    @Test
    public void testRemoveIndex() {
        LongArrayList list = new LongArrayList();
        list.add(0).add(1).add(2).add(3);
        Assert.assertEquals(1, list.remove(1));
        Assert.assertEquals(3, list.size());
        Assert.assertArrayEquals(new long[] {0, 2, 3}, list.toArray());
        Assert.assertEquals(3, list.remove(2));
        Assert.assertArrayEquals(new long[] {0, 2}, list.toArray());
        Assert.assertEquals(0, list.remove(0));
        Assert.assertArrayEquals(new long[] {2}, list.toArray());
        list.add(0).add(1);
        Assert.assertEquals(3, list.size());
        Assert.assertArrayEquals(new long[] {2, 0, 1}, list.toArray());
    }

}
