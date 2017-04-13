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

import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.Arrays;

import org.apache.hadoop.hive.ql.metadata.HiveException;
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
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topic 2 -alpha 0.00001 -delta 0.00001")};

        udtf.initialize(argOIs);

        String[] doc1 = new String[]{"fruits:1", "healthy:1", "vegetables:1"};
        String[] doc2 = new String[]{"apples:1", "avocados:1", "colds:1", "flu:1", "like:2", "oranges:1"};
        for (int it = 0; it < 10000; it++) {
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
            udtf.getProbability("vegetables", k1) > udtf.getProbability("flu", k1));
        Assert.assertTrue("doc2 is in topic " + k2 + " (" + (topicDistr[k2] * 100) + "%), "
            + "and `avocados` SHOULD be more suitable topic word than `healthy` in the topic",
            udtf.getProbability("avocados", k2) > udtf.getProbability("healthy", k2));
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
