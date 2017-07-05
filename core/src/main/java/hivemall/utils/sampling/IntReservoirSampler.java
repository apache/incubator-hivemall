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
package hivemall.utils.sampling;

import java.util.Arrays;
import java.util.Random;

import javax.annotation.Nonnull;

/**
 * Vitter's reservoir sampling implementation that randomly chooses k items from a list containing n
 * items.
 * 
 * @link http://en.wikipedia.org/wiki/Reservoir_sampling
 * @link http://portal.acm.org/citation.cfm?id=3165
 */
public final class IntReservoirSampler {

    private final int[] samples;
    private final int numSamples;
    private int position;

    private final Random rand;

    public IntReservoirSampler(int sampleSize) {
        if (sampleSize <= 0) {
            throw new IllegalArgumentException("sampleSize must be greater than 1: " + sampleSize);
        }
        this.samples = new int[sampleSize];
        this.numSamples = sampleSize;
        this.position = 0;
        this.rand = new Random();
    }

    public IntReservoirSampler(int sampleSize, long seed) {
        this.samples = new int[sampleSize];
        this.numSamples = sampleSize;
        this.position = 0;
        this.rand = new Random(seed);
    }

    public IntReservoirSampler(int[] samples) {
        this.samples = samples;
        this.numSamples = samples.length;
        this.position = 0;
        this.rand = new Random();
    }

    public IntReservoirSampler(int[] samples, long seed) {
        this.samples = samples;
        this.numSamples = samples.length;
        this.position = 0;
        this.rand = new Random(seed);
    }

    public int size() {
        return position;
    }

    @Nonnull
    public int[] getSample() {
        if (position >= numSamples) {
            return samples;
        }
        return Arrays.copyOf(samples, position);
    }

    public void add(final int item) {
        if (position < numSamples) {// reservoir not yet full, just append
            samples[position] = item;
        } else {// find a item to replace
            int replaceIndex = rand.nextInt(position + 1);
            if (replaceIndex < numSamples) {
                samples[replaceIndex] = item;
            }
        }
        position++;
    }

    public void clear() {
        Arrays.fill(samples, 0);
        this.position = 0;
    }
}
