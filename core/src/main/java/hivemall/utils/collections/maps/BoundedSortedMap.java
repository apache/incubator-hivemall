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

import hivemall.utils.lang.Preconditions;

import java.util.Collections;
import java.util.Map.Entry;
import java.util.TreeMap;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnegative;
import javax.annotation.Nullable;

public final class BoundedSortedMap<K, V> extends TreeMap<K, V> {
    private static final long serialVersionUID = 4580890152997313541L;

    private final int bound;

    public BoundedSortedMap(@Nonnegative int size) {
        this(size, false);
    }

    public BoundedSortedMap(@Nonnegative int size, boolean reverseOrder) {
        super(reverseOrder ? Collections.reverseOrder() : null);
        Preconditions.checkArgument(size > 0, "size must be greater than zero: " + size);
        this.bound = size;
    }

    @Nullable
    public V put(@CheckForNull final K key, @Nullable final V value) {
        final V old = super.put(key, value);
        if (size() > bound) {
            Entry<K, V> e = pollLastEntry();
            if (e == null) {
                return null;
            }
            return e.getValue();
        }
        return old;
    }

}
