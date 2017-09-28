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
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;

import javax.annotation.Nonnull;
import java.util.List;

public final class SkipGramModel extends AbstractWord2VecModel {

    protected SkipGramModel(final int dim, final int win, final int neg, final int iter,
            final float startingLR, final long numTrainWords, final Int2FloatOpenHashTable S,
            final int[] aliasWordId) {
        super(dim, win, neg, iter, startingLR, numTrainWords, S, aliasWordId);
    }

    protected void trainOnDoc(@Nonnull final int[] doc) {
        final int vecDim = dim;
        final int numNegative = neg;
        final PRNG rnd = _rnd;

        // alias sampler for negative sampling
        final Int2FloatOpenHashTable S = _S;
        final int[] aliasWordId = _aliasWordId;

        // reuse variable
        int windowSize, k, targetWord, inputWord, contextWord;
        float label, gradient;

        updateLearningRate();

        final int docLength = doc.length;
        for (int t = 0; t < iter; t++) {
            for (int inputWordPosition = 0; inputWordPosition < docLength; inputWordPosition++) {
                inputWord = doc[inputWordPosition];

                if (!inputWeights.containsKey(inputWord * vecDim)) {
                    initWordWeights(inputWord);
                }

                windowSize = rnd.nextInt(win) + 1;

                for (int contextPosition = inputWordPosition - windowSize; contextPosition < inputWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == inputWordPosition || contextPosition < 0
                            || contextPosition >= docLength) {
                        continue;
                    }

                    contextWord = doc[contextPosition];
                    final float[] gradVec = new float[vecDim];

                    // negative sampling
                    for (int d = 0; d < numNegative + 1; d++) {
                        // positive
                        if (d == 0) {
                            targetWord = contextWord;
                            label = 1.f;
                        } else {
                            do {
                                k = rnd.nextInt(S.size());

                                if (S.get(k) > rnd.nextDouble()) {
                                    targetWord = k;
                                } else {
                                    targetWord = aliasWordId[k];
                                }
                            } while (targetWord == contextWord);
                            label = 0.f;
                        }

                        // update context vector
                        gradient = grad(label, inputWord, targetWord) * lr;
                        for (int i = 0; i < vecDim; i++) {
                            gradVec[i] += gradient * contextWeights.get(targetWord * vecDim + i);
                            contextWeights.put(targetWord * vecDim + i,
                                contextWeights.get(targetWord * vecDim + i) + gradient
                                        * inputWeights.get(inputWord * vecDim + i));
                        }
                    }

                    // update inWord vector
                    for (int i = 0; i < vecDim; i++) {
                        inputWeights.put(inputWord * vecDim + i,
                            inputWeights.get(inputWord * vecDim + i) + gradVec[i]);
                    }
                }
            }
        }

        wordCount += docLength * iter;
    }

    protected float grad(final float label, final int w, final int c) {
        float dotValue = 0.f;
        for (int i = 0; i < dim; i++) {
            dotValue += inputWeights.get(w * dim + i) * contextWeights.get(c * dim + i);
        }

        return (label - sigmoid(dotValue, sigmoidTable));
    }
}
