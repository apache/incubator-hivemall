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

import javax.annotation.Nonnull;

public final class SkipGramModel extends AbstractWord2vecModel {

    protected SkipGramModel(final int dim, final float startingLR, final long numTrainWords) {
        super(dim, startingLR, numTrainWords);
    }

    protected void onlineTrain(final int inWord, final int posWord, @Nonnull final int[] negWords) {

        final int vecDim = dim;

        updateLearningRate();

        if (!inputWeights.containsKey(inWord * vecDim)) {
            initWordWeights(inWord);
        }

        float[] gradVec = new float[vecDim];

        // positive words
        float gradient = grad(1.f, inWord, posWord) * lr;
        for (int i = 0; i < vecDim; i++) {
            gradVec[i] += gradient * contextWeights.get(posWord * vecDim + i);
            contextWeights.put(
                posWord * vecDim + i,
                gradient * inputWeights.get(inWord * vecDim + i)
                        + contextWeights.get(posWord * vecDim + i));
        }

        // negative words
        for (int negWord : negWords) {
            gradient = grad(0.f, inWord, negWord) * lr;
            for (int i = 0; i < vecDim; i++) {
                gradVec[i] += gradient * contextWeights.get(negWord * vecDim + i);
                contextWeights.put(
                    negWord * vecDim + i,
                    gradient * inputWeights.get(inWord * vecDim + i)
                            + contextWeights.get(negWord * vecDim + i));
            }
        }

        // update inWord vector
        for (int i = 0; i < vecDim; i++) {
            inputWeights.put(inWord * vecDim + i,
                gradVec[i] + inputWeights.get(inWord * vecDim + i));
        }

        wordCount++;
    }
}
