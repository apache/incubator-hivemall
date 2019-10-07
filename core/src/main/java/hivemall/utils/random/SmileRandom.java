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

import smile.math.random.RandomNumberGenerator;
import smile.math.random.UniversalGenerator;

public final class SmileRandom implements PRNG {

    @Nonnull
    private RandomNumberGenerator rng;

    public SmileRandom() {
        this.rng = new UniversalGenerator();
    }

    public SmileRandom(long seed) {
        this.rng = new UniversalGenerator(seed);
    }

    public SmileRandom(@Nonnull RandomNumberGenerator rng) {
        this.rng = rng;
    }

    @Override
    public int nextInt(int n) {
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
