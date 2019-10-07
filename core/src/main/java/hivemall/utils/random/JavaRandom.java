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
package hivemall.utils.random;

import java.util.Random;

import javax.annotation.Nonnull;

public final class JavaRandom implements PRNG {

    private final Random rand;

    public JavaRandom() {
        this.rand = new Random();
    }

    public JavaRandom(long seed) {
        this.rand = new Random(seed);
    }

    public JavaRandom(@Nonnull Random rand) {
        this.rand = rand;
    }

    @Override
    public int nextInt(int n) {
        return rand.nextInt(n);
    }

    @Override
    public int nextInt() {
        return rand.nextInt();
    }

    @Override
    public long nextLong() {
        return rand.nextLong();
    }

    @Override
    public double nextDouble() {
        return rand.nextDouble();
    }

}
