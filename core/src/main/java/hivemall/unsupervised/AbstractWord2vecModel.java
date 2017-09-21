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

import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;

import javax.annotation.Nonnull;

public abstract class AbstractWord2vecModel {
    // cached sigmoid function parameters
    private final int MAX_SIGMOID = 6;
    private final int SIGMOID_TABLE_SIZE = 1000;
    private Int2FloatOpenHashTable sigmoidTable;

    // learning rate parameters
    protected float lr;
    private float startingLR;
    private long numTrainWords;
    protected long wordCount;
    private long lastWordCount;
    private long wordCountActual;

    protected int dim;
    private PRNG _rnd;

    protected Int2FloatOpenHashTable contextWeights;
    protected Int2FloatOpenHashTable inputWeights;

    protected AbstractWord2vecModel(final int dim, final float startingLR, final long numTrainWords) {
        this.dim = dim;
        this.startingLR = this.lr = startingLR;
        this.numTrainWords = numTrainWords;

        this.wordCount = 0L;
        this.lastWordCount = 0L;
        this.wordCountActual = 0L;
        this._rnd = RandomNumberGeneratorFactory.createPRNG(1001);

        this.sigmoidTable = initSigmoidTable(MAX_SIGMOID, SIGMOID_TABLE_SIZE);

        // TODO how to estimate size
        this.inputWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.inputWeights.defaultReturnValue(0.f);
        this.contextWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.contextWeights.defaultReturnValue(0.f);
    }

    private static Int2FloatOpenHashTable initSigmoidTable(final int maxSigmoid,
            final int sigmoidTableSize) {
        Int2FloatOpenHashTable sigmoidTable = new Int2FloatOpenHashTable(sigmoidTableSize);
        for (int i = 0; i < sigmoidTableSize; i++) {
            float x = ((float) i / sigmoidTableSize * 2 - 1) * (float) maxSigmoid;
            sigmoidTable.put(i, 1.f / ((float) Math.exp(-x) + 1.f));
        }
        return sigmoidTable;
    }

    protected void initWordWeights(final int wordId) {
        for (int i = 0; i < dim; i++) {
            inputWeights.put(wordId * dim + i, ((float) _rnd.nextDouble() - 0.5f) / dim);
        }
    }

    // cannot use for CBoW
    protected float grad(final float label, final int w, final int c) {
        float dotValue = 0.f;
        for (int i = 0; i < dim; i++) {
            dotValue += inputWeights.get(w * dim + i) * contextWeights.get(c * dim + i);
        }

        return (label - sigmoid(dotValue, MAX_SIGMOID, SIGMOID_TABLE_SIZE, sigmoidTable));
    }

    private static float sigmoid(final float v, final int MAX_SIGMOID,
            final int SIGMOID_TABLE_SIZE, final Int2FloatOpenHashTable sigmoidTable) {
        if (v > MAX_SIGMOID) {
            return 1.f;
        } else if (v < -MAX_SIGMOID) {
            return 0.f;
        } else {
            return sigmoidTable.get((int) ((v + MAX_SIGMOID) * (SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)));
        }
    }

    protected void updateLearningRate() {
        // TODO: valid lr?

        if (wordCount - lastWordCount > 10000) {
            wordCountActual += wordCount - lastWordCount;
            lastWordCount = wordCount;

            this.lr = Math.max(startingLR * (1.f - (float) wordCountActual / (numTrainWords + 1L)),
                startingLR * 0.0001f);
        }
    }

    protected abstract void onlineTrain(final int inWord, final int posWord,
            @Nonnull final int[] negWords);
}
