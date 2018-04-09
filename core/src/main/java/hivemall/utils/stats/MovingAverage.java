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
package hivemall.utils.stats;

import hivemall.utils.collections.DoubleRingBuffer;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Preconditions;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class MovingAverage {

    @Nonnull
    private final DoubleRingBuffer ring;

    private double totalSum;

    public MovingAverage(@Nonnegative int windowSize) {
        Preconditions.checkArgument(windowSize > 1, "Invalid window size: " + windowSize);
        this.ring = new DoubleRingBuffer(windowSize);
        this.totalSum = 0.d;
    }

    public double add(final double x) {
        if (!NumberUtils.isFinite(x)) {
            throw new IllegalArgumentException("Detected Infinite input: " + x);
        }

        if (ring.isFull()) {
            double head = ring.head();
            this.totalSum -= head;
        }
        ring.add(x);
        totalSum += x;

        final int size = ring.size();
        if (size == 0) {
            return 0.d;
        }
        return totalSum / size;
    }

    public double get() {
        final int size = ring.size();
        if (size == 0) {
            return 0.d;
        }
        return totalSum / size;
    }

    @Override
    public String toString() {
        return "MovingAverage [ring=" + ring + ", total=" + totalSum + ", moving_avg=" + get()
                + "]";
    }

}
