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
package hivemall.embedding;

import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public abstract class AbstractWord2VecModel {
    // cached sigmoid function parameters
    protected static final int MAX_SIGMOID = 6;
    protected static final int SIGMOID_TABLE_SIZE = 1000;
    protected float[] sigmoidTable;

    // learning rate parameters
    @Nonnegative
    protected float lr;
    @Nonnegative
    private float startingLR;
    @Nonnegative
    private long numTrainWords;
    @Nonnegative
    protected long wordCount;
    @Nonnegative
    private long lastWordCount;
    @Nonnegative
    private long wordCountActual;

    @Nonnegative
    protected int dim;
    private PRNG rnd;

    protected Int2FloatOpenHashTable contextWeights;
    protected Int2FloatOpenHashTable inputWeights;

    protected AbstractWord2VecModel(final int dim, final float startingLR, final long numTrainWords) {
        this.dim = dim;
        this.startingLR = this.lr = startingLR;
        this.numTrainWords = numTrainWords;

        this.wordCount = 0L;
        this.lastWordCount = 0L;
        this.wordCountActual = 0L;
        this.rnd = RandomNumberGeneratorFactory.createPRNG(1001);

        this.sigmoidTable = initSigmoidTable();

        // TODO how to estimate size
        this.inputWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.inputWeights.defaultReturnValue(0.f);
        this.contextWeights = new Int2FloatOpenHashTable(10578 * dim);
        this.contextWeights.defaultReturnValue(0.f);
    }

    private static float[] initSigmoidTable() {
        float[] sigmoidTable = new float[SIGMOID_TABLE_SIZE];
        for (int i = 0; i < SIGMOID_TABLE_SIZE; i++) {
            float x = ((float) i / SIGMOID_TABLE_SIZE * 2 - 1) * (float) MAX_SIGMOID;
            sigmoidTable[i] = 1.f / ((float) Math.exp(-x) + 1.f);
        }
        return sigmoidTable;
    }

    protected void initWordWeights(final int wordId) {
        for (int i = 0; i < dim; i++) {
            inputWeights.put(wordId * dim + i, ((float) rnd.nextDouble() - 0.5f) / dim);
        }
    }

    protected static float sigmoid(final float v, final float[] sigmoidTable) {
        if (v > MAX_SIGMOID) {
            return 1.f;
        } else if (v < -MAX_SIGMOID) {
            return 0.f;
        } else {
            return sigmoidTable[(int) ((v + MAX_SIGMOID) * (SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2))];
        }
    }

    protected void updateLearningRate() {
        // TODO: valid lr?

        if (wordCount - lastWordCount > 10000) {
            wordCountActual += wordCount - lastWordCount;
            lastWordCount = wordCount;

            this.lr = startingLR
                    * Math.max((1.f - (float) wordCountActual / (numTrainWords + 1L)), 0.0001f);
        }
    }

    protected abstract void onlineTrain(final int inWord, final int posWord,
            @Nonnull final int[] negWords);

    protected abstract void onlineTrain(final int[] inWords, final int posWord,
            @Nonnull final int[] negWords);
}
