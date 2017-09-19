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
package hivemall.unsupervised;

import hivemall.utils.collections.maps.Int2FloatOpenHashTable;

import javax.annotation.Nonnull;
import java.util.Random;

public abstract class AbstractWord2vecModel {
    protected final int maxSigmoid = 6;
    protected final int sigmoidTableSize = 1000;
    protected Int2FloatOpenHashTable sigmoidTable;

    protected int dim;
    private Random rnd;

    protected Int2FloatOpenHashTable contextWeights;
    protected Int2FloatOpenHashTable inputWeights;

    public AbstractWord2vecModel(final int dim) {
        this.dim = dim;
        this.rnd = new Random();

        this.sigmoidTable = initSigmoidTable(maxSigmoid, sigmoidTableSize);

        // TODO how to estimate size
        this.inputWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.inputWeights.defaultReturnValue(0.f);
        this.contextWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.contextWeights.defaultReturnValue(0.f);
    }

    protected static Int2FloatOpenHashTable initSigmoidTable(final double maxSigmoid,
            final int sigmoidTableSize) {
        Int2FloatOpenHashTable sigmoidTable = new Int2FloatOpenHashTable(sigmoidTableSize);
        for (int i = 0; i < sigmoidTableSize; i++) {
            float x = ((float) i / sigmoidTableSize * 2 - 1) * (float) maxSigmoid;
            sigmoidTable.put(i, 1.f / ((float) Math.exp(-x) + 1.f));
        }
        return sigmoidTable;
    }

    protected float grad(final int label, final int w, final int c) {
        if (!inputWeights.containsKey(w)) {
            for (int i = 0; i < dim; i++) {
                inputWeights.put(w * dim + i, (this.rnd.nextFloat() - 0.5f) / this.dim);
            }
        }

        float dotValue = 0.f;
        for (int i = 0; i < dim; i++) {
            dotValue += inputWeights.get(w * dim + i) * contextWeights.get(c * dim + i);
        }

        return (label - sigmoid(dotValue));
    }

    private float sigmoid(final float v) {
        if (v > maxSigmoid) {
            return 1.f;
        } else if (v < -maxSigmoid) {
            return 0.f;
        } else {
            return sigmoidTable.get((int) ((v + maxSigmoid) * (sigmoidTableSize / maxSigmoid / 2)));
        }
    }

    protected static float getLearningRate(@Nonnull final long wordCountActual,
            @Nonnull final long numTrainWords, @Nonnull final float startingLR) {
        return Math.max(startingLR * (1.f - (float) wordCountActual / (numTrainWords + 1L)),
            startingLR * 0.0001f);
    }

    protected abstract void onlineTrain(@Nonnull final int inWord, @Nonnull final int posWord,
            @Nonnull final int[] negWords, @Nonnull final float lr);
}
