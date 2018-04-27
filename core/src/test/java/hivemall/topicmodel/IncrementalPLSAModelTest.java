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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.SortedMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import hivemall.classifier.KernelExpansionPassiveAggressiveUDTFTest;

import org.junit.Assert;
import org.junit.Test;

import javax.annotation.Nonnull;

public class IncrementalPLSAModelTest {
    private static final boolean DEBUG = false;

    @Test
    public void testOnline() {
        int K = 2;
        int it = 0;
        int maxIter = 1024;
        float perplexityPrev;
        float perplexity = Float.MAX_VALUE;

        IncrementalPLSAModel model = new IncrementalPLSAModel(K, 0.5f, 1E-5d);

        String[] doc1 = new String[] {"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 =
                new String[] {"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};

        do {
            perplexityPrev = perplexity;
            perplexity = 0.f;

            // online (i.e., one-by-one) updating
            model.train(new String[][] {doc1});
            perplexity += model.computePerplexity();

            model.train(new String[][] {doc2});
            perplexity += model.computePerplexity();

            perplexity /= 2.f; // mean perplexity for the 2 docs

            it++;
            println("Iteration " + it + ": mean perplexity = " + perplexity);
        } while (it < maxIter && Math.abs(perplexityPrev - perplexity) >= 1E-4f);

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
        Assert.assertTrue(
            "doc1 is in topic " + k1 + " (" + (topicDistr[k1] * 100) + "%), "
                    + "and `vegetables` SHOULD be more suitable topic word than `flu` in the topic",
            model.getWordScore("vegetables", k1) > model.getWordScore("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
                + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            model.getWordScore("avocados", k2) > model.getWordScore("healthy", k2));
    }

    @Test
    public void testMiniBatch() {
        int K = 2;
        int it = 0;
        int maxIter = 2048;
        float perplexityPrev;
        float perplexity = Float.MAX_VALUE;

        IncrementalPLSAModel model = new IncrementalPLSAModel(K, 0.5f, 1E-5d);

        String[] doc1 = new String[] {"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 =
                new String[] {"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};

        do {
            perplexityPrev = perplexity;

            model.train(new String[][] {doc1, doc2});
            perplexity = model.computePerplexity();

            it++;
            println("Iteration " + it + ": perplexity = " + perplexity);
        } while (it < maxIter && Math.abs(perplexityPrev - perplexity) >= 1E-4f);

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
        Assert.assertTrue(
            "doc1 is in topic " + k1 + " (" + (topicDistr[k1] * 100) + "%), "
                    + "and `vegetables` SHOULD be more suitable topic word than `flu` in the topic",
            model.getWordScore("vegetables", k1) > model.getWordScore("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
                + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            model.getWordScore("avocados", k2) > model.getWordScore("healthy", k2));
    }

    @Test
    public void testNews20() throws IOException {
        int K = 20;
        int miniBatchSize = 2;

        int cnt, it;
        int maxIter = 64;

        IncrementalPLSAModel model = new IncrementalPLSAModel(K, 100.f, 1E-3d);

        BufferedReader news20 = readFile("news20-multiclass.gz");

        String[][] docs = new String[K][];

        String line = news20.readLine();
        List<String> doc = new ArrayList<String>();

        cnt = 0;
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");

            int k = Integer.parseInt(tokens.nextToken()) - 1;

            while (tokens.hasMoreTokens()) {
                doc.add(tokens.nextToken());
            }

            // store first document in each of K classes
            if (docs[k] == null) {
                docs[k] = doc.toArray(new String[doc.size()]);
                cnt++;
            }

            if (cnt == K) {
                break;
            }

            doc.clear();
            line = news20.readLine();
        }
        println("Stored " + cnt + " docs. Start training w/ mini-batch size: " + miniBatchSize);

        float perplexityPrev;
        float perplexity = Float.MAX_VALUE;

        it = 0;
        do {
            perplexityPrev = perplexity;
            perplexity = 0.f;

            int head = 0;
            cnt = 0;
            while (head < K) {
                int tail = head + miniBatchSize;
                model.train(Arrays.copyOfRange(docs, head, tail));
                perplexity += model.computePerplexity();
                head = tail;
                cnt++;
                println("Processed mini-batch#" + cnt);
            }

            perplexity /= cnt;

            it++;
            println("Iteration " + it + ": mean perplexity = " + perplexity);
        } while (it < maxIter && Math.abs(perplexityPrev - perplexity) >= 1E-3f);

        Set<Integer> topics = new HashSet<Integer>();
        for (int k = 0; k < K; k++) {
            topics.add(findMaxTopic(model.getTopicDistribution(docs[k])));
        }

        int n = topics.size();
        println("# of unique topics: " + n);
        Assert.assertTrue("At least 15 documents SHOULD be classified to different topics, "
                + "but there are only " + n + " unique topics.",
            n >= 15);
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        // use data stored for KPA UDTF test
        InputStream is =
                KernelExpansionPassiveAggressiveUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    @Nonnull
    private static int findMaxTopic(@Nonnull float[] topicDistr) {
        int maxIdx = 0;
        for (int i = 1; i < topicDistr.length; i++) {
            if (topicDistr[maxIdx] < topicDistr[i]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

}
