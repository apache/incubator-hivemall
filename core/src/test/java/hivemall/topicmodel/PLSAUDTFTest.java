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

import hivemall.TestUtils;
import hivemall.utils.lang.mutable.MutableInt;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class PLSAUDTFTest {
    private static final boolean DEBUG = false;

    @Test
    public void test() throws HiveException {
        PLSAUDTF udtf = new PLSAUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-topics 2 -alpha 0.1 -delta 0.00001 -iter 10000")};

        udtf.initialize(argOIs);

        String[] doc1 = new String[] {"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 =
                new String[] {"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};

        udtf.process(new Object[] {Arrays.asList(doc1)});
        udtf.process(new Object[] {Arrays.asList(doc2)});

        udtf.finalizeTraining();

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

        Assert.assertTrue(
            "doc1 is in topic " + k1 + " (" + (topicDistr[k1] * 100) + "%), "
                    + "and `vegetables` SHOULD be more suitable topic word than `flu` in the topic",
            udtf.getWordScore("vegetables", k1) > udtf.getWordScore("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
                + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            udtf.getWordScore("avocados", k2) > udtf.getWordScore("healthy", k2));
    }

    @Test
    public void testMultiBytes() throws HiveException {
        PLSAUDTF udtf = new PLSAUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-topics 2 -alpha 0.1 -delta 0.00001 -iter 10000 -mini_batch_size 1")};

        udtf.initialize(argOIs);

        String[] doc1 = new String[] {"果物:1", "健康:1", "野菜:1"};
        String[] doc2 = new String[] {"りんご:1", "アボカド:1", "風邪:1", "インフルエンザ:1", "好き:2", "みかん:1"};

        udtf.process(new Object[] {Arrays.asList(doc1)});
        udtf.process(new Object[] {Arrays.asList(doc2)});

        udtf.finalizeTraining();

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

        Assert.assertTrue(
            "doc1 is in topic " + k1 + " (" + (topicDistr[k1] * 100) + "%), "
                    + "and `野菜` SHOULD be more suitable topic word than `インフルエンザ` in the topic",
            udtf.getWordScore("野菜", k1) > udtf.getWordScore("インフルエンザ", k1));
        Assert.assertTrue(
            "doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
                    + "and `アボカド` SHOULD be more suitable topic word than `健康` in the topic",
            udtf.getWordScore("アボカド", k2) > udtf.getWordScore("健康", k2));
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(PLSAUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-topics 2 -alpha 0.1 -delta 0.00001 -iter 10000")},
            new Object[][] {{Arrays.asList("fruits:1", "healthy:1", "vegetables:1")},
                    {Arrays.asList("apples:1", "avocados:1", "colds:1", "flu:1", "like:2",
                        "oranges:1")}});
    }

    @Test
    public void testSingleRow() throws HiveException {
        PLSAUDTF udtf = new PLSAUDTF();
        final int numTopics = 2;
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-topics " + numTopics)};
        udtf.initialize(argOIs);

        String[] doc1 = new String[] {"1", "2", "3"};
        udtf.process(new Object[] {Arrays.asList(doc1)});

        final MutableInt cnt = new MutableInt(0);
        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object arg0) throws HiveException {
                cnt.addValue(1);
            }
        });
        udtf.close();

        Assert.assertEquals(doc1.length * numTopics, cnt.getValue());
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
