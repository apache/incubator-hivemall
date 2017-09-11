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
package hivemall.utils.hashing;

public final class HashUtils {

    private HashUtils() {}

    public static int jenkins32(int k) {
        k = (k + 0x7ed55d16) + (k << 12);
        k = (k ^ 0xc761c23c) ^ (k >> 19);
        k = (k + 0x165667b1) + (k << 5);
        k = (k + 0xd3a2646c) ^ (k << 9);
        k = (k + 0xfd7046c5) + (k << 3);
        k = (k ^ 0xb55a4f09) ^ (k >> 16);
        return k;
    }

    public static int murmurHash3(int k) {
        k ^= k >>> 16;
        k *= 0x85ebca6b;
        k ^= k >>> 13;
        k *= 0xc2b2ae35;
        k ^= k >>> 16;
        return k;
    }

    public static int fnv1a(final int k) {
        int hash = 0x811c9dc5;
        for (int i = 0; i < 4; i++) {
            hash ^= k << (i * 8);
            hash *= 0x01000193;
        }
        return hash;
    }

    /**
     * https://gist.github.com/badboy/6267743
     */
    public static int hash32shift(int k) {
        k = ~k + (k << 15); // key = (key << 15) - key - 1;
        k = k ^ (k >>> 12);
        k = k + (k << 2);
        k = k ^ (k >>> 4);
        k = k * 2057; // key = (key + (key << 3)) + (key << 11);
        k = k ^ (k >>> 16);
        return k;
    }

    public static int hash32shiftmult(int k) {
        k = (k ^ 61) ^ (k >>> 16);
        k = k + (k << 3);
        k = k ^ (k >>> 4);
        k = k * 0x27d4eb2d;
        k = k ^ (k >>> 15);
        return k;
    }

    /**
     * http://burtleburtle.net/bob/hash/integer.html
     */
    public static int hash7shifts(int k) {
        k -= (k << 6);
        k ^= (k >> 17);
        k -= (k << 9);
        k ^= (k << 4);
        k -= (k << 3);
        k ^= (k << 10);
        k ^= (k >> 15);
        return k;
    }

}
