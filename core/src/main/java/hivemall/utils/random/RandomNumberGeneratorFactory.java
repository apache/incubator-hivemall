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

import hivemall.utils.lang.Primitives;

import java.security.SecureRandom;

import javax.annotation.Nonnull;

public final class RandomNumberGeneratorFactory {

    private RandomNumberGeneratorFactory() {}

    @Nonnull
    public static PRNG createPRNG() {
        return createPRNG(PRNGType.smile);
    }

    @Nonnull
    public static PRNG createPRNG(long seed) {
        return createPRNG(PRNGType.smile, seed);
    }

    @Nonnull
    public static PRNG createPRNG(@Nonnull PRNGType type) {
        final PRNG rng;
        switch (type) {
            case java:
                rng = new JavaRandom();
                break;
            case secure:
                rng = new JavaRandom(new SecureRandom());
                break;
            case smile:
                rng = new SmileRandom();
                break;
            case smileMT:
                rng = new SmileRandom(new smile.math.random.MersenneTwister());
                break;
            case smileMT64:
                rng = new SmileRandom(new smile.math.random.MersenneTwister64());
                break;
            case commonsMath3MT:
                rng = new CommonsMathRandom(new org.apache.commons.math3.random.MersenneTwister());
                break;
            default:
                throw new IllegalStateException("Unexpected type: " + type);
        }
        return rng;
    }

    @Nonnull
    public static PRNG createPRNG(@Nonnull PRNGType type, long seed) {
        final PRNG rng;
        switch (type) {
            case java:
                rng = new JavaRandom(seed);
                break;
            case secure:
                rng = new JavaRandom(new SecureRandom(Primitives.toBytes(seed)));
                break;
            case smile:
                rng = new SmileRandom(seed);
                break;
            case smileMT:
                rng = new SmileRandom(
                    new smile.math.random.MersenneTwister(Primitives.hashCode(seed)));
                break;
            case smileMT64:
                rng = new SmileRandom(new smile.math.random.MersenneTwister64(seed));
                break;
            case commonsMath3MT:
                rng = new CommonsMathRandom(
                    new org.apache.commons.math3.random.MersenneTwister(seed));
                break;
            default:
                throw new IllegalStateException("Unexpected type: " + type);
        }
        return rng;
    }

    public enum PRNGType {
        java, secure, smile, smileMT, smileMT64, commonsMath3MT;
    }

}
