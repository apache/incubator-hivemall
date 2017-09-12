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
package hivemall.utils.collections;

import hivemall.utils.lang.NaturalComparator;
import hivemall.utils.lang.StringUtils;

import java.util.Collections;
import java.util.Comparator;

import org.junit.Assert;
import org.junit.Test;

public class BoundedPriorityQueueTest {

    @Test
    public void testTop3() {
        BoundedPriorityQueue<Integer> queue = new BoundedPriorityQueue<Integer>(3,
            new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return Integer.compare(o1, o2);
                }
            });
        Assert.assertTrue(queue.offer(1));
        Assert.assertTrue(queue.offer(4));
        Assert.assertTrue(queue.offer(3));
        Assert.assertTrue(queue.offer(2));
        Assert.assertFalse(queue.offer(1));
        Assert.assertTrue(queue.offer(2));
        Assert.assertTrue(queue.offer(3));

        Assert.assertEquals(3, queue.size());

        Assert.assertEquals(Integer.valueOf(3), queue.peek());
        Assert.assertEquals(Integer.valueOf(3), queue.poll());
        Assert.assertEquals(Integer.valueOf(3), queue.poll());
        Assert.assertEquals(Integer.valueOf(4), queue.poll());
        Assert.assertNull(queue.poll());
        Assert.assertEquals(0, queue.size());
    }

    @Test
    public void testTail3() {
        BoundedPriorityQueue<Integer> queue = new BoundedPriorityQueue<Integer>(3,
            Collections.<Integer>reverseOrder());
        Assert.assertTrue(queue.offer(1));
        Assert.assertTrue(queue.offer(4));
        Assert.assertTrue(queue.offer(3));
        Assert.assertTrue(queue.offer(2));
        Assert.assertTrue(queue.offer(1));
        Assert.assertTrue(queue.offer(2));
        Assert.assertFalse(queue.offer(3));

        Assert.assertEquals(3, queue.size());

        Assert.assertEquals(Integer.valueOf(2), queue.peek());
        Assert.assertEquals(Integer.valueOf(2), queue.poll());
        Assert.assertEquals(Integer.valueOf(1), queue.poll());
        Assert.assertEquals(Integer.valueOf(1), queue.poll());
        Assert.assertNull(queue.poll());
        Assert.assertEquals(0, queue.size());
    }

    @Test
    public void testString1() {
        BoundedPriorityQueue<String> queue = new BoundedPriorityQueue<>(3,
            new Comparator<String>() {
                @Override
                public int compare(String o1, String o2) {
                    return StringUtils.compare(o1, o2);
                }
            });
        queue.offer("B");
        queue.offer("A");
        queue.offer("C");
        queue.offer("D");
        Assert.assertEquals("B", queue.poll());
        Assert.assertEquals("C", queue.poll());
        Assert.assertEquals("D", queue.poll());
        Assert.assertNull(queue.poll());
    }

    @Test
    public void testString2() {
        BoundedPriorityQueue<String> queue = new BoundedPriorityQueue<>(3,
            NaturalComparator.<String>getInstance());
        queue.offer("B");
        queue.offer("A");
        queue.offer("C");
        queue.offer("D");
        Assert.assertEquals("B", queue.poll());
        Assert.assertEquals("C", queue.poll());
        Assert.assertEquals("D", queue.poll());
        Assert.assertNull(queue.poll());
    }

}
