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
package hivemall.topicmodel;

import java.util.Map;
import java.util.List;
import java.util.SortedMap;

import org.junit.Assert;
import org.junit.Test;

public class OnlineLDAModelTest {
    private static final boolean DEBUG = false;

    @Test
    public void test() {
        int K = 2;
        int it = 0;
        float perplexityPrev;
        float perplexity = Float.MAX_VALUE;

        OnlineLDAModel model = new OnlineLDAModel(K, 1.f / K, 1.f / K, 2, 80, 0.8, 1E-5d);

        String[] doc1 = new String[] {"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 = new String[] {"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};
        String[][] miniBatch = new String[][] {doc1, doc2};

        do {
            // online (i.e., one-by-one) updating
            model.train(miniBatch);

            it++;
            perplexityPrev = perplexity;
            perplexity = model.computePerplexity();
            println("Iteration " + it + ": perplexity = " + perplexity);
        } while(Math.abs(perplexityPrev - perplexity) >= 1E-6f);

        SortedMap<Float, List<String>> topicWords;

        println("Topic 0:");
        println("========");
        topicWords = model.getTopicWords(0);
        for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
            List<String> words = e.getValue();
            for (int i = 0; i < words.size(); i++) {
                println(e.getKey() + " " + words.get(i));
            }
        }
        println("========");

        println("Topic 1:");
        println("========");
        topicWords = model.getTopicWords(1);
        for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
            List<String> words = e.getValue();
            for (int i = 0; i < words.size(); i++) {
                println(e.getKey() + " " + words.get(i));
            }
        }
        println("========");

        int k1, k2;
        float[] topicDistr = model.getTopicDistribution(doc1);
        if (topicDistr[0] > topicDistr[1]) {
            // topic 0 MUST represent doc#1
            k1 = 0;
            k2 = 1;
        } else {
            k1 = 1;
            k2 = 0;
        }
        Assert.assertTrue("doc1 is in topic " + k1 + " (" + (topicDistr[k1] * 100) + "%), "
            + "and `vegetables` SHOULD be more suitable topic word than `flu` in the topic",
            model.getLambda("vegetables", k1) > model.getLambda("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
            + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            model.getLambda("avocados", k2) > model.getLambda("healthy", k2));
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

}
