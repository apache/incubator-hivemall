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
import hivemall.utils.collections.maps.Int2IntOpenHashTable;

import javax.annotation.Nonnull;
import java.util.List;

public final class SkipGramModel extends AbstractWord2vecModel {

    public SkipGramModel(int dim, int win, int neg, long numTrainWords, Int2FloatOpenHashTable S,
            Int2IntOpenHashTable A) {
        super(dim, win, neg, numTrainWords, S, A);
    }

    protected void iteration(@Nonnull final List<Integer> doc, @Nonnull final float lr) {
        int docLength = doc.size();
        for (int inputWordPosition = 0; inputWordPosition < docLength; inputWordPosition++) {
            int w = doc.get(inputWordPosition);
            int windowSize = rnd.nextInt(win);
            float[] gradVec = new float[dim];

            for (int contextPosition = inputWordPosition - windowSize; contextPosition < inputWordPosition
                    + windowSize + 1; contextPosition++) {
                if (contextPosition == inputWordPosition)
                    continue;
                if (contextPosition < 0)
                    continue;
                if (contextPosition >= docLength)
                    continue;
                int c = doc.get(contextPosition);

                int target, label;
                for (int d = 0; d < neg + 1; d++) {
                    if (d == 0) {
                        target = c;
                        label = 1;
                    } else {
                        target = negativeSample(c);
                        label = 0;
                    }
                    float grad = grad(label, w, target) * lr;

                    for (int i = 0; i < dim; i++) {
                        gradVec[i] += grad * contextWeights.get(target * dim + i);
                        this.contextWeights.put(
                            target * dim + i,
                            grad * inputWeights.get(w * dim + i)
                                    + contextWeights.get(target * dim + i));
                    }
                }

                for (int i = 0; i < dim; i++) {
                    this.inputWeights.put(w * dim + i,
                        gradVec[i] + this.inputWeights.get(w * dim + i));
                }
            }
        }
    }
}
