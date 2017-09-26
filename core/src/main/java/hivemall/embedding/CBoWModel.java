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

public final class CBoWModel extends AbstractWord2VecModel {
    protected CBoWModel(final int dim, final int win, final int neg, final int iter,
            final float startingLR, final long numTrainWords, final Int2FloatOpenHashTable S,
            final int[] aliasWordId) {
        super(dim, win, neg, iter, startingLR, numTrainWords, S, aliasWordId);
    }

    protected void trainOnDoc(@Nonnull List<Integer> doc) {
        final int vecDim = dim;
        final int numNegative = neg;
        final PRNG _rnd = rnd;
        final Int2FloatOpenHashTable _S = S;
        final int[] _aliasWordId = aliasWordId;
        float label, gradient;

        // reuse instance
        int windowSize, k, numContext, targetWord, inWord, positiveWord;

        updateLearningRate();

        int docLength = doc.size();
        for (int t = 0; t < iter; t++) {
            for (int positiveWordPosition = 0; positiveWordPosition < docLength; positiveWordPosition++) {
                windowSize = _rnd.nextInt(win) + 1;

                numContext = windowSize * 2 + Math.min(0, positiveWordPosition - windowSize)
                        + Math.min(0, docLength - positiveWordPosition - windowSize - 1);

                float[] gradVec = new float[vecDim];
                float[] averageVec = new float[vecDim];

                // collect context words
                for (int contextPosition = positiveWordPosition - windowSize; contextPosition < positiveWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == positiveWordPosition || contextPosition < 0
                            || contextPosition >= docLength) {
                        continue;
                    }

                    inWord = doc.get(contextPosition);

                    // average vector of input word vectors
                    if (!inputWeights.containsKey(inWord * vecDim)) {
                        initWordWeights(inWord);
                    }

                    for (int i = 0; i < vecDim; i++) {
                        averageVec[i] += inputWeights.get(inWord * vecDim + i) / numContext;
                    }
                }
                positiveWord = doc.get(positiveWordPosition);
                // negative sampling
                for (int d = 0; d < numNegative + 1; d++) {
                    if (d == 0) {
                        targetWord = positiveWord;
                        label = 1.f;
                    } else {
                        do {
                            k = _rnd.nextInt(_S.size());
                            if (_S.get(k) > _rnd.nextDouble()) {
                                targetWord = k;
                            } else {
                                targetWord = _aliasWordId[k];
                            }
                        } while (targetWord == positiveWord);
                        label = 0.f;
                    }

                    gradient = grad(label, averageVec, targetWord) * lr;
                    for (int i = 0; i < vecDim; i++) {
                        gradVec[i] += gradient * contextWeights.get(targetWord * vecDim + i);
                        contextWeights.put(targetWord * vecDim + i,
                            contextWeights.get(targetWord * vecDim + i) + gradient * averageVec[i]);
                    }
                }

                // update inWord vector
                for (int contextPosition = positiveWordPosition - windowSize; contextPosition < positiveWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == positiveWordPosition || contextPosition < 0
                            || contextPosition >= docLength) {
                        continue;
                    }

                    inWord = doc.get(contextPosition);
                    for (int i = 0; i < vecDim; i++) {
                        inputWeights.put(inWord * vecDim + i, inputWeights.get(inWord * vecDim + i)
                                + gradVec[i]);
                    }
                }
            }
        }

        wordCount += docLength * iter;
    }

    private float grad(final float label, final float[] w, final int c) {
        float dotValue = 0.f;
        for (int i = 0; i < dim; i++) {
            dotValue += w[i] * contextWeights.get(c * dim + i);
        }

        return (label - sigmoid(dotValue, sigmoidTable));
    }
}
