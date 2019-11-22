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

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public final class LongCounter<E> implements Serializable {
    private static final long serialVersionUID = 7949630590734361716L;

    private final Map<E, Long> counts;

    public LongCounter() {
        this.counts = new HashMap<E, Long>();
    }

    public LongCounter(@Nonnull Map<E, Long> counts) {
        this.counts = counts;
    }

    public Map<E, Long> getMap() {
        return counts;
    }

    public long increment(E key) {
        return increment(key, 1L);
    }

    public long increment(E key, long amount) {
        Long count = counts.get(key);
        if (count == null) {
            counts.put(key, Long.valueOf(amount));
            return 0;
        } else {
            long old = count.longValue();
            counts.put(key, Long.valueOf(old + amount));
            return old;
        }
    }

    public long getCount(E key) {
        Long count = counts.get(key);
        if (count == null) {
            return 0;
        } else {
            return count.longValue();
        }
    }

    public void addAll(Map<E, Long> counter) {
        if (counter == null) {
            return;
        }
        for (Map.Entry<E, Long> e : counter.entrySet()) {
            increment(e.getKey(), e.getValue().longValue());
        }
    }

    public void addAll(LongCounter<E> counter) {
        if (counter == null) {
            return;
        }
        for (Map.Entry<E, Long> e : counter.entrySet()) {
            increment(e.getKey(), e.getValue().longValue());
        }
    }

    public Set<Map.Entry<E, Long>> entrySet() {
        return counts.entrySet();
    }

    @Nullable
    public E whichMax() {
        E maxKey = null;
        long maxValue = Long.MIN_VALUE;
        for (Map.Entry<E, Long> e : counts.entrySet()) {
            long v = e.getValue().longValue();
            if (v >= maxValue) {
                maxValue = v;
                maxKey = e.getKey();
            }
        }
        return maxKey;
    }

    @Nullable
    public E whichMin() {
        E minKey = null;
        long minValue = Long.MAX_VALUE;
        for (Map.Entry<E, Long> e : counts.entrySet()) {
            long v = e.getValue().longValue();
            if (v <= minValue) {
                minValue = v;
                minKey = e.getKey();
            }
        }
        return minKey;
    }

    public int size() {
        return counts.size();
    }

}
