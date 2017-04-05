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

import java.util.Map;
import java.util.SortedMap;

import org.junit.Assert;
import org.junit.Test;

public class OnlineLDAModelTest {
    private static final boolean DEBUG = true;

    @Test
    public void test() {
        int K = 2;
        int numIter = 20;

        OnlineLDAModel model = new OnlineLDAModel(K, 1.f / K, 1.f / K, 2, 80, 0.8, 1E-5d);

        for (int it = 0; it < numIter; it++) {
            // online (i.e., one-by-one) updating
            model.train(new String[][] {new String[] {"fruits:1", "healthy:1", "vegetables:1"}}, 1L);

            model.train(new String[][] {new String[] {"apples:1", "avocados:1", "colds:1", "flu:1",
                    "like:2", "oranges:1"}}, 2L);

            println("Iteration " + it + ": perplexity = " + model.computePerplexity());
        }

        SortedMap<Float, String> topicWords;

        println("Topic 0:");
        println("========");
        topicWords = model.getTopicWords(0);
        for (Map.Entry<Float, String> e : topicWords.entrySet()) {
            println(e.getKey() + " " + e.getValue());
        }
        println("========");

        println("Topic 1:");
        println("========");
        topicWords = model.getTopicWords(1);
        for (Map.Entry<Float, String> e : topicWords.entrySet()) {
            println(e.getKey() + " " + e.getValue());
        }
        println("========");


        int k1, k2;
        if (model.getLambda("fruits", 0) > model.getLambda("apples", 0)) {
            // topic 0 MUST represent doc#1
            k1 = 0;
            k2 = 1;
        } else {
            k1 = 1;
            k2 = 0;
        }
        Assert.assertTrue("`vegetables` SHOULD be more suitable topic word than `flu` in topic " + k1,
            model.getLambda("vegetables", k1) > model.getLambda("flu", k1));
        Assert.assertTrue("`avocados` SHOULD be more suitable topic word than `healthy` in topic " + k2,
            model.getLambda("avocados", k2) > model.getLambda("healthy", k2));
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

}
