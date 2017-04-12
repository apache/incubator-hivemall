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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.SortedMap;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;
import java.text.ParseException;

import hivemall.classifier.KernelExpansionPassiveAggressiveUDTFTest;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Assert;
import org.junit.Test;

import javax.annotation.Nonnull;

public class LDAUDTFTest {
    private static final boolean DEBUG = false;

    @Test
    public void test() throws HiveException {
        LDAUDTF udtf = new LDAUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
            ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
            ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topic 2 -iter 20")};

        udtf.initialize(argOIs);

        String[] doc1 = new String[]{"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 = new String[]{"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};
        for (int it = 0; it < 20; it++) {
            udtf.process(new Object[]{ Arrays.asList(doc1) });
            udtf.process(new Object[]{ Arrays.asList(doc2) });
        }

        SortedMap<Float, List<String>> topicWords;

        println("Topic 0:");
        println("========");
        topicWords = udtf.getTopicWords(0);
        for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
            List<String> words = e.getValue();
            for (int i = 0; i < words.size(); i++) {
                println(e.getKey() + " " + words.get(i));
            }
        }
        println("========");

        println("Topic 1:");
        println("========");
        topicWords = udtf.getTopicWords(1);
        for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
            List<String> words = e.getValue();
            for (int i = 0; i < words.size(); i++) {
                println(e.getKey() + " " + words.get(i));
            }
        }
        println("========");

        int k1, k2;
        float[] topicDistr = udtf.getTopicDistribution(doc1);
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
            udtf.getLambda("vegetables", k1) > udtf.getLambda("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
            + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            udtf.getLambda("avocados", k2) > udtf.getLambda("healthy", k2));
    }

    @Test
    public void testNews20() throws IOException, ParseException, HiveException {
        LDAUDTF udtf = new LDAUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topic 20 -delta 0.1")};

        udtf.initialize(argOIs);

        BufferedReader news20 = readFile("news20-small.binary.gz");

        List<String> doc = new ArrayList<String>();

        String[] docInClass1 = new String[0];
        String[] docInClass2 = new String[0];

        String line = news20.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            int label = Integer.parseInt(tokens.nextToken());

            while (tokens.hasMoreTokens()) {
                doc.add(tokens.nextToken());
            }

            udtf.process(new Object[]{ doc });

            if (docInClass1.length == 0 && label == 1) { // store first +1 document
                docInClass1 = doc.toArray(new String[doc.size()]);
            } else if (docInClass2.length == 0 && label == -1) { // store first -1 document
                docInClass2 = doc.toArray(new String[doc.size()]);
            }

            doc.clear();
            line = news20.readLine();
        }

        SortedMap<Float, List<String>> topicWords;

        for (int k = 0; k < 20; k++) {
            println("========");
            println("Topic " + k);
            topicWords = udtf.getTopicWords(k, 5);
            for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
                List<String> words = e.getValue();
                for (int i = 0; i < words.size(); i++) {
                    println(e.getKey() + " " + words.get(i));
                }
            }
            println("========");
        }

        int k1 = findMaxTopic(udtf.getTopicDistribution(docInClass1));
        int k2 = findMaxTopic(udtf.getTopicDistribution(docInClass2));
        Assert.assertTrue("Two documents which are respectively in class#1 (+1) and #2 (-1) are assigned to the same topic: "
            + k1 + ". Documents in the different class SHOULD be assigned to the different topics.", k1 != k2);
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        // use data stored for KPA UDTF test
        InputStream is = KernelExpansionPassiveAggressiveUDTFTest.class.getResourceAsStream(fileName);
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
