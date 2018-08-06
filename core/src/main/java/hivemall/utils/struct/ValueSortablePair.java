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

import hivemall.utils.lang.Preconditions;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;

public final class ValueSortablePair<K, V extends Comparable<V>>
        implements Comparable<ValueSortablePair<K, V>> {

    @Nonnull
    private final K k;
    @Nonnull
    private final V v;

    public ValueSortablePair(@CheckForNull K k, @Nonnull V v) {
        this.k = Preconditions.checkNotNull(k);
        this.v = Preconditions.checkNotNull(v);
    }

    @Nonnull
    public K getKey() {
        return k;
    }

    @Nonnull
    public V getValue() {
        return v;
    }

    @Override
    public int compareTo(ValueSortablePair<K, V> o) {
        return v.compareTo(o.v);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + k.hashCode();
        result = prime * result + v.hashCode();
        return result;
    }

    @SuppressWarnings("unchecked")
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        ValueSortablePair<K, V> other = (ValueSortablePair<K, V>) obj;
        if (!k.equals(other.k))
            return false;
        if (!v.equals(other.v))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return "k=" + k + ", v=" + v;
    }

}
