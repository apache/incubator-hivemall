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
package hivemall.lda;

import org.junit.Test;

public class OnlineLDAModelTest {

    @Test
    public void testRunability() {
        int K = 2;
        int batchSize = 1; // purely online setting
        int numIter = 20;

        OnlineLDAModel model = new OnlineLDAModel(K,
            1.d / K, 1.d / K, 2,80, 0.8, batchSize);

        for (int it = 0; it < numIter; it++) {
            // online (i.e., one-by-one) updating
            model.train(new String[][]{
                new String[]{"fruits:1", "healthy:1", "vegetables:1"}}, 1);

            model.train(new String[][]{
                new String[]{"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"}}, 2);

            System.out.println("Iteration " + it + ": perplexity = " + model.getPerplexity());
        }

        model.showTopicWords();
    }

}
