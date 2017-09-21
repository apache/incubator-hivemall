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

    protected void onlineTrain(@Nonnull final int inWord, @Nonnull final int posWord,
            @Nonnull final int[] negWords) {

        updateLearningRate();

        // initialized weights for inWord
        if (!inputWeights.containsKey(inWord * dim)) {
            for (int i = 0; i < dim; i++) {
                inputWeights.put(inWord * dim + i, ((float) _rnd.nextDouble() - 0.5f) / dim);
            }
        }

        float[] gradVec = new float[dim];

        // positive
        float gradient = grad(1.f, inWord, posWord) * lr;
        for (int i = 0; i < dim; i++) {
            gradVec[i] += gradient * contextWeights.get(posWord * dim + i);
            contextWeights.put(posWord * dim + i, gradient * inputWeights.get(inWord * dim + i)
                    + contextWeights.get(posWord * dim + i));
        }

        // negative
        for (int target : negWords) {
            gradient = grad(0.f, inWord, target) * lr;
            for (int i = 0; i < dim; i++) {
                gradVec[i] += gradient * contextWeights.get(target * dim + i);
                contextWeights.put(target * dim + i, gradient * inputWeights.get(inWord * dim + i)
                        + contextWeights.get(target * dim + i));
            }
        }

        // update inWord vector
        for (int i = 0; i < dim; i++) {
            inputWeights.put(inWord * dim + i, gradVec[i] + inputWeights.get(inWord * dim + i));
        }

        wordCount++;
    }
}
