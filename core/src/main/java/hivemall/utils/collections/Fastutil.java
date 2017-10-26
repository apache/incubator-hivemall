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

import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2LongMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectMap;
import it.unimi.dsi.fastutil.objects.ObjectIterable;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;

import javax.annotation.Nonnull;

/**
 * Helper class for fastutil (http://fastutil.di.unimi.it/)
 */
public final class Fastutil {

    private Fastutil() {}

    @Nonnull
    public static ObjectIterable<Int2LongMap.Entry> fastIterable(@Nonnull final Int2LongMap map) {
        final ObjectSet<Int2LongMap.Entry> entries = map.int2LongEntrySet();
        return entries instanceof Int2LongMap.FastEntrySet
                ? new ObjectIterable<Int2LongMap.Entry>() {
                    public ObjectIterator<Int2LongMap.Entry> iterator() {
                        return ((Int2LongMap.FastEntrySet) entries).fastIterator();
                    }
                }
                : entries;
    }

    @Nonnull
    public static ObjectIterable<Int2FloatMap.Entry> fastIterable(@Nonnull final Int2FloatMap map) {
        final ObjectSet<Int2FloatMap.Entry> entries = map.int2FloatEntrySet();
        return entries instanceof Int2FloatMap.FastEntrySet
                ? new ObjectIterable<Int2FloatMap.Entry>() {
                    public ObjectIterator<Int2FloatMap.Entry> iterator() {
                        return ((Int2FloatMap.FastEntrySet) entries).fastIterator();
                    }
                }
                : entries;
    }

    @Nonnull
    public static <V> ObjectIterable<Int2ObjectMap.Entry<V>> fastIterable(
            @Nonnull final Int2ObjectMap<V> map) {
        final ObjectSet<Int2ObjectMap.Entry<V>> entries = map.int2ObjectEntrySet();
        return entries instanceof Int2ObjectMap.FastEntrySet
                ? new ObjectIterable<Int2ObjectMap.Entry<V>>() {
                    public ObjectIterator<Int2ObjectMap.Entry<V>> iterator() {
                        return ((Int2ObjectMap.FastEntrySet<V>) entries).fastIterator();
                    }
                }
                : entries;
    }

    @Nonnull
    public static <K, V> ObjectIterable<Object2ObjectMap.Entry<K, V>> fastIterable(
            @Nonnull final Object2ObjectMap<K, V> map) {
        final ObjectSet<Object2ObjectMap.Entry<K, V>> entries = map.object2ObjectEntrySet();
        return entries instanceof Object2ObjectMap.FastEntrySet
                ? new ObjectIterable<Object2ObjectMap.Entry<K, V>>() {
                    @SuppressWarnings("unchecked")
                    public ObjectIterator<Object2ObjectMap.Entry<K, V>> iterator() {
                        return ((Object2ObjectMap.FastEntrySet<K, V>) entries).fastIterator();
                    }
                }
                : entries;
    }

}
