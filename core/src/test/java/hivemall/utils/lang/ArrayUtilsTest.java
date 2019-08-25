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

import org.junit.Assert;
import org.junit.Test;

public class ArrayUtilsTest {

    @Test
    public void testSortedArraySet() {
        final int[] original = new int[] {3, 7, 10};
        Assert.assertSame(original, ArrayUtils.sortedArraySet(original, 7));
        Assert.assertSame(original, ArrayUtils.sortedArraySet(original, 3));
        Assert.assertSame(original, ArrayUtils.sortedArraySet(original, 10));
        Assert.assertArrayEquals(new int[] {3, 7, 8, 10},
            ArrayUtils.sortedArraySet(new int[] {3, 7, 10}, 8));
        Assert.assertArrayEquals(new int[] {3, 7, 7, 8, 10},
            ArrayUtils.sortedArraySet(new int[] {3, 7, 7, 10}, 8));
        Assert.assertArrayEquals(new int[] {3, 7, 7, 10},
            ArrayUtils.sortedArraySet(new int[] {3, 7, 7, 10}, 7));
        Assert.assertArrayEquals(new int[] {3, 7, 10, 11},
            ArrayUtils.sortedArraySet(new int[] {3, 7, 10}, 11));
        Assert.assertArrayEquals(new int[] {-2, 3, 7, 10},
            ArrayUtils.sortedArraySet(new int[] {3, 7, 10}, -2));
    }

    @Test
    public void testAppendIntArrayInt() {
        Assert.assertArrayEquals(new int[] {3, 7, 10, 8},
            ArrayUtils.append(new int[] {3, 7, 10}, 8));
    }

    @Test
    public void testInsert() {
        final int[] original = new int[] {3, 7, 10};
        Assert.assertArrayEquals(new int[] {3, 7, 8, 10}, ArrayUtils.insert(original, 2, 8));
        Assert.assertArrayEquals(new int[] {1, 3, 7, 10}, ArrayUtils.insert(original, 0, 1));
        Assert.assertArrayEquals(new int[] {3, 3, 7, 10}, ArrayUtils.insert(original, 0, 3));
        Assert.assertArrayEquals(new int[] {3, 3, 7, 10}, ArrayUtils.insert(original, 0, 3));
        Assert.assertArrayEquals(new int[] {3, 7, 10, 11},
            ArrayUtils.insert(original, original.length, 11));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInsertFail() {
        final int[] original = new int[] {3, 7, 10};
        Assert.assertArrayEquals(new int[] {3, 7, 10, 11},
            ArrayUtils.insert(original, original.length + 1, 11));
    }

}
