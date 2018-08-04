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

public class KeySortableValueTest {

    @Test
    public void test1() {
        KeySortableValue<Float, Integer> v1 = new KeySortableValue<>(1.f, -1);
        KeySortableValue<Float, Integer> v2 = new KeySortableValue<>(2.f, 3);
        KeySortableValue<Float, Integer> v3 = new KeySortableValue<>(3.f, 2);
        KeySortableValue<Float, Integer> v4 = new KeySortableValue<>(4.f, 0);

        PriorityQueue<KeySortableValue<Float, Integer>> pq =
                new PriorityQueue<>(11, Collections.reverseOrder());
        pq.add(v1);
        pq.add(v2);
        pq.add(v3);
        pq.add(v4);

        assertEquals(3, pq.poll().getValue().intValue());
        assertEquals(2, pq.poll().getValue().intValue());
        assertEquals(0, pq.poll().getValue().intValue());
        assertEquals(-1, pq.poll().getValue().intValue());

        assertTrue(pq.isEmpty());
    }

    @SuppressWarnings("unchecked")
    @Test
    public void test2() {
        KeySortableValue<Float, Integer> v1 = new KeySortableValue<>(1.f, -1);
        KeySortableValue<Float, Integer> v2 = new KeySortableValue<>(2.f, 3);
        KeySortableValue<Float, Integer> v3 = new KeySortableValue<>(3.f, 2);
        KeySortableValue<Float, Integer> v4 = new KeySortableValue<>(4.f, 0);

        KeySortableValue<Float, Integer>[] arr = new KeySortableValue[] {v1, v2, v3, v4};
        Arrays.sort(arr, Collections.reverseOrder());

        assertEquals(3, arr[0].getValue().intValue());
        assertEquals(2, arr[1].getValue().intValue());
        assertEquals(0, arr[2].getValue().intValue());
        assertEquals(-1, arr[3].getValue().intValue());
    }

}
