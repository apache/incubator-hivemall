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

import javax.annotation.Nonnull;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public final class CommonsMathRandom implements PRNG {

    @Nonnull
    private final RandomGenerator rng;

    public CommonsMathRandom() {
        this.rng = new MersenneTwister();
    }

    public CommonsMathRandom(long seed) {
        this.rng = new MersenneTwister(seed);
    }

    public CommonsMathRandom(@Nonnull RandomGenerator rng) {
        this.rng = rng;
    }

    @Override
    public int nextInt(final int n) {
        return rng.nextInt(n);
    }

    @Override
    public int nextInt() {
        return rng.nextInt();
    }

    @Override
    public long nextLong() {
        return rng.nextLong();
    }

    @Override
    public double nextDouble() {
        return rng.nextDouble();
    }

}
