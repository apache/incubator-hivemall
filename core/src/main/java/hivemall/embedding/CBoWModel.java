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

import javax.annotation.Nonnull;

public final class CBoWModel extends AbstractWord2VecModel {

    protected CBoWModel(final int dim, final float startingLR, final long numTrainWords) {
        super(dim, startingLR, numTrainWords);
    }

    protected void onlineTrain(final int inWord, final int posWord, @Nonnull final int[] negWords) {
        throw new UnsupportedOperationException();
    }

    protected void onlineTrain(final int[] inWords, final int posWord, @Nonnull final int[] negWords) {

        final int vecDim = dim;

        updateLearningRate();

        float[] gradVec = new float[vecDim];
        float[] averageVec = new float[vecDim];

        // average vector of input word vectors
        for (int inWord : inWords) {
            if (!inputWeights.containsKey(inWord * vecDim)) {
                initWordWeights(inWord);
            }

            for (int i = 0; i < vecDim; i++) {
                averageVec[i] += inputWeights.get(inWord * vecDim + i) / inWords.length;
            }
        }

        // positive word
        float gradient = grad(1.f, averageVec, posWord) * lr;
        for (int i = 0; i < vecDim; i++) {
            gradVec[i] += gradient * contextWeights.get(posWord * vecDim + i);
            contextWeights.put(posWord * vecDim + i, contextWeights.get(posWord * vecDim + i)
                    + gradient * averageVec[i]);
        }

        // negative words
        for (int negWord : negWords) {
            gradient = grad(0.f, averageVec, negWord) * lr;
            for (int i = 0; i < vecDim; i++) {
                gradVec[i] += gradient * contextWeights.get(negWord * vecDim + i);
                contextWeights.put(negWord * vecDim + i, contextWeights.get(negWord * vecDim + i)
                        + gradient * averageVec[i]);
            }
        }

        // update inWord vector
        for (int inWord : inWords) {
            for (int i = 0; i < vecDim; i++) {
                inputWeights.put(inWord * vecDim + i, inputWeights.get(inWord * vecDim + i)
                        + gradVec[i]);
            }
        }

        wordCount++;
    }

    private float grad(final float label, final float[] w, final int c) {
        float dotValue = 0.f;
        for (int i = 0; i < dim; i++) {
            dotValue += w[i] * contextWeights.get(c * dim + i);
        }

        return (label - sigmoid(dotValue, sigmoidTable));
    }
}
