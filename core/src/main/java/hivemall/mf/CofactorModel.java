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

import hivemall.math.matrix.sparse.DoKMatrix;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class CofactorModel extends FactorizedModel {

    private static final int EXPECTED_SIZE = 136861;
    private Int2ObjectMap<Rating[]> contextItems;
    private Int2ObjectMap<Rating> contextBias;
    private DoKMatrix cooccurMatrix;

    // hyperparameters
    private double c0, c1;

    public CofactorModel(@Nonnull RatingInitializer ratingInitializer, @Nonnegative int factor,
                         @Nonnull RankInitScheme initScheme, @Nonnull int numItems,
                         @Nonnull double c0, @Nonnull int c1) {

        // rank init scheme is gaussian
        // https://github.com/dawenl/cofactor/blob/master/src/cofacto.py#L98
        super(ratingInitializer, factor, 0.f, initScheme, EXPECTED_SIZE);
        this.contextItems = new Int2ObjectOpenHashMap<Rating[]>(EXPECTED_SIZE);
        this.contextBias = new Int2ObjectOpenHashMap<Rating>(EXPECTED_SIZE);
        this.cooccurMatrix = new DoKMatrix(numItems, numItems);
        checkHyperparameterC(c0);
        checkHyperparameterC(c1);
        this.c0 = c0;
        this.c1 = c1;
    }

    @Nullable
    public Rating[] getContextVector(final int c) {
        return getContextVector(c, false);
    }

    @Nullable
    public Rating[] getContextVector(final int c, final boolean init) {
        Rating[] v = contextItems.get(c);
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
            contextItems.put(c, v);
            this.maxIndex = Math.max(maxIndex, c);
            this.minIndex = Math.min(minIndex, c);
        }
        return v;
    }

    @Nonnull
    public Rating contextBias(final int c) {
        Rating b = contextBias.get(c);
        if (b == null) {
            b = ratingInitializer.newRating(0.f); // dummy
            contextBias.put(c, b);
        }
        return b;
    }

    public float getContextBias(final int c) {
        final Rating b = contextBias.get(c);
        if (b == null) {
            return 0.f;
        }
        return b.getWeight();
    }

    public void setContextBias(final int c, final float value) {
        Rating b = contextBias.get(c);
        if (b == null) {
            b = ratingInitializer.newRating(value);
            contextBias.put(c, b);
        }
        b.setWeight(value);
    }

    public void incrementCooccurrence(final int row, final int col) {
         final double count = this.cooccurMatrix.get(row, col, 0.d);
         this.cooccurMatrix.set(row, col, count + 1);
    }

    private static void checkHyperparameterC(final double c) {
        assert c >= 0.d && c <= 1.d;
    }
}
