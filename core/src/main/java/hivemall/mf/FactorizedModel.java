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
package hivemall.mf;

import hivemall.utils.math.MathUtils;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

import java.util.Random;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.annotation.concurrent.NotThreadSafe;

@NotThreadSafe
public class FactorizedModel {

    @Nonnull
    protected final RatingInitializer ratingInitializer;
    @Nonnegative
    protected final int factor;

    // rank matrix initialization
    protected final RankInitScheme initScheme;

    protected int minIndex, maxIndex;
    @Nonnull
    private Rating meanRating;
    private Int2ObjectMap<Rating[]> users;
    private Int2ObjectMap<Rating[]> items;
    private Int2ObjectMap<Rating> userBias;
    private Int2ObjectMap<Rating> itemBias;

    protected final Random[] randU, randI;

    public FactorizedModel(@Nonnull RatingInitializer ratingInitializer, @Nonnegative int factor,
            @Nonnull RankInitScheme initScheme) {
        this(ratingInitializer, factor, 0.f, initScheme, 136861);
    }

    public FactorizedModel(@Nonnull RatingInitializer ratingInitializer, @Nonnegative int factor,
            float meanRating, @Nonnull RankInitScheme initScheme) {
        this(ratingInitializer, factor, meanRating, initScheme, 136861);
    }

    public FactorizedModel(@Nonnull RatingInitializer ratingInitializer, @Nonnegative int factor,
            float meanRating, @Nonnull RankInitScheme initScheme, int expectedSize) {
        this.ratingInitializer = ratingInitializer;
        this.factor = factor;
        this.initScheme = initScheme;
        this.minIndex = 0;
        this.maxIndex = 0;
        this.meanRating = ratingInitializer.newRating(meanRating);
        this.users = new Int2ObjectOpenHashMap<Rating[]>(expectedSize);
        this.items = new Int2ObjectOpenHashMap<Rating[]>(expectedSize);
        this.userBias = new Int2ObjectOpenHashMap<Rating>(expectedSize);
        this.itemBias = new Int2ObjectOpenHashMap<Rating>(expectedSize);
        this.randU = newRandoms(factor, 31L);
        this.randI = newRandoms(factor, 41L);
    }

    public enum RankInitScheme {
        random /* default */, gaussian;

        @Nonnegative
        protected float maxInitValue;
        @Nonnegative
        protected double initStdDev;

        @Nonnull
        public static RankInitScheme resolve(@Nullable String opt) {
            if (opt == null) {
                return random;
            } else if ("gaussian".equalsIgnoreCase(opt)) {
                return gaussian;
            } else if ("random".equalsIgnoreCase(opt)) {
                return random;
            }
            return random;
        }

        public void setMaxInitValue(float maxInitValue) {
            this.maxInitValue = maxInitValue;
        }

        public void setInitStdDev(double initStdDev) {
            this.initStdDev = initStdDev;
        }

    }

    @Nonnull
    private static Random[] newRandoms(@Nonnull final int size, final long seed) {
        final Random[] rand = new Random[size];
        for (int i = 0, len = rand.length; i < len; i++) {
            rand[i] = new Random(seed + i);
        }
        return rand;
    }

    public int getMinIndex() {
        return minIndex;
    }

    public int getMaxIndex() {
        return maxIndex;
    }

    @Nonnull
    public Rating meanRating() {
        return meanRating;
    }

    public float getMeanRating() {
        return meanRating.getWeight();
    }

    public void setMeanRating(final float rating) {
        meanRating.setWeight(rating);
    }

    @Nullable
    public Rating[] getUserVector(final int u) {
        return getUserVector(u, false);
    }

    @Nullable
    public Rating[] getUserVector(final int u, final boolean init) {
        Rating[] v = users.get(u);
        if (init && v == null) {
            v = new Rating[factor];
            switch (initScheme) {
                case random:
                    uniformFill(v, randU[0], initScheme.maxInitValue, ratingInitializer);
                    break;
                case gaussian:
                    gaussianFill(v, randU, initScheme.initStdDev, ratingInitializer);
                    break;
                default:
                    throw new IllegalStateException(
                        "Unsupported rank initialization scheme: " + initScheme);

            }
            users.put(u, v);
            this.maxIndex = Math.max(maxIndex, u);
            this.minIndex = Math.min(minIndex, u);
        }
        return v;
    }

    @Nullable
    public Rating[] getItemVector(final int i) {
        return getItemVector(i, false);
    }

    @Nullable
    public Rating[] getItemVector(int i, boolean init) {
        Rating[] v = items.get(i);
        if (init && v == null) {
            v = new Rating[factor];
            switch (initScheme) {
                case random:
                    uniformFill(v, randI[0], initScheme.maxInitValue, ratingInitializer);
                    break;
                case gaussian:
                    gaussianFill(v, randI, initScheme.initStdDev, ratingInitializer);
                    break;
                default:
                    throw new IllegalStateException(
                        "Unsupported rank initialization scheme: " + initScheme);

            }
            items.put(i, v);
            this.maxIndex = Math.max(maxIndex, i);
            this.minIndex = Math.min(minIndex, i);
        }
        return v;
    }

    @Nonnull
    public Rating userBias(final int u) {
        Rating b = userBias.get(u);
        if (b == null) {
            b = ratingInitializer.newRating(0.f); // dummy
            userBias.put(u, b);
        }
        return b;
    }

    public float getUserBias(final int u) {
        final Rating b = userBias.get(u);
        if (b == null) {
            return 0.f;
        }
        return b.getWeight();
    }

    public void setUserBias(final int u, final float value) {
        Rating b = userBias.get(u);
        if (b == null) {
            b = ratingInitializer.newRating(value);
            userBias.put(u, b);
        }
        b.setWeight(value);
    }

    @Nonnull
    public Rating itemBias(final int i) {
        Rating b = itemBias.get(i);
        if (b == null) {
            b = ratingInitializer.newRating(0.f); // dummy
            itemBias.put(i, b);
        }
        return b;
    }

    @Nullable
    public Rating getItemBiasObject(final int i) {
        return itemBias.get(i);
    }

    public float getItemBias(final int i) {
        final Rating b = itemBias.get(i);
        if (b == null) {
            return 0.f;
        }
        return b.getWeight();
    }

    public void setItemBias(final int i, final float value) {
        Rating b = itemBias.get(i);
        if (b == null) {
            b = ratingInitializer.newRating(value);
            itemBias.put(i, b);
        }
        b.setWeight(value);
    }

    protected static void uniformFill(final Rating[] a, final Random rand, final float maxInitValue,
            final RatingInitializer init) {
        for (int i = 0, len = a.length; i < len; i++) {
            float v = rand.nextFloat() * maxInitValue / len;
            a[i] = init.newRating(v);
        }
    }

    protected static void gaussianFill(final Rating[] a, final Random[] rand, final double stddev,
            final RatingInitializer init) {
        for (int i = 0, len = a.length; i < len; i++) {
            float v = (float) MathUtils.gaussian(0.d, stddev, rand[i]);
            a[i] = init.newRating(v);
        }
    }

}
