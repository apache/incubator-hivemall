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
package hivemall.utils.struct;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;

import org.junit.Test;

public class KeySortablePairTest {

    @Test
    public void testPriorityQueue() {
        KeySortablePair<Float, Integer> v1 = new KeySortablePair<>(3.f, 1);
        KeySortablePair<Float, Integer> v2 = new KeySortablePair<>(1.f, 2);
        KeySortablePair<Float, Integer> v3 = new KeySortablePair<>(4.f, 3);
        KeySortablePair<Float, Integer> v4 = new KeySortablePair<>(-1.f, 4);

        PriorityQueue<KeySortablePair<Float, Integer>> pq =
                new PriorityQueue<>(11, Collections.reverseOrder());
        pq.add(v1);
        pq.add(v2);
        pq.add(v3);
        pq.add(v4);

        assertEquals(Float.valueOf(4.f), pq.poll().getKey());
        assertEquals(Float.valueOf(3.f), pq.poll().getKey());
        assertEquals(Float.valueOf(1.f), pq.poll().getKey());
        assertEquals(Float.valueOf(-1.f), pq.poll().getKey());

        assertTrue(pq.isEmpty());
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testArraySort() {
        KeySortablePair<Float, Integer> v1 = new KeySortablePair<>(3.f, 1);
        KeySortablePair<Float, Integer> v2 = new KeySortablePair<>(1.f, 2);
        KeySortablePair<Float, Integer> v3 = new KeySortablePair<>(4.f, 3);
        KeySortablePair<Float, Integer> v4 = new KeySortablePair<>(-1.f, 4);

        KeySortablePair<Float, Integer>[] arr = new KeySortablePair[] {v1, v2, v3, v4};
        Arrays.sort(arr, Collections.reverseOrder());

        assertEquals(Float.valueOf(4.f), arr[0].getKey());
        assertEquals(Float.valueOf(3.f), arr[1].getKey());
        assertEquals(Float.valueOf(1.f), arr[2].getKey());
        assertEquals(Float.valueOf(-1.f), arr[3].getKey());
    }

}
