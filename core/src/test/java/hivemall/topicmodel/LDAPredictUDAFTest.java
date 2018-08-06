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

import hivemall.utils.math.MathUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.SimpleGenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class LDAPredictUDAFTest {
    LDAPredictUDAF udaf;
    GenericUDAFEvaluator evaluator;
    ObjectInspector[] inputOIs;
    ObjectInspector[] partialOI;
    LDAPredictUDAF.OnlineLDAPredictAggregationBuffer agg;

    String[] words;
    int[] labels;
    float[] lambdas;

    @Before
    public void setUp() throws Exception {
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

        fieldNames.add("wcList");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector));

        fieldNames.add("lambdaMap");
        fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaFloatObjectInspector)));

        fieldNames.add("topics");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

        fieldNames.add("alpha");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        fieldNames.add("delta");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        partialOI = new ObjectInspector[4];
        partialOI[0] =
                ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);

        words = new String[] {"fruits", "vegetables", "healthy", "flu", "apples", "oranges", "like",
                "avocados", "colds", "colds", "avocados", "oranges", "like", "apples", "flu",
                "healthy", "vegetables", "fruits"};
        labels = new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        lambdas = new float[] {0.3339331f, 0.3324783f, 0.33209667f, 3.2804057E-4f, 3.0303953E-4f,
                2.4860457E-4f, 2.41481E-4f, 2.3554532E-4f, 1.352576E-4f, 0.1660153f, 0.16596903f,
                0.1659654f, 0.1659627f, 0.16593699f, 0.1659259f, 0.0017611005f, 0.0015791848f,
                8.84464E-4f};
    }

    @Test
    public void test() throws Exception {
        udaf = new LDAPredictUDAF();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.STRING),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.INT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topics 2")};

        evaluator = udaf.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        agg = (LDAPredictUDAF.OnlineLDAPredictAggregationBuffer) evaluator.getNewAggregationBuffer();

        final Map<String, Float> doc1 = new HashMap<String, Float>();
        doc1.put("fruits", 1.f);
        doc1.put("healthy", 1.f);
        doc1.put("vegetables", 1.f);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            evaluator.iterate(agg, new Object[] {word, doc1.get(word), labels[i], lambdas[i]});
        }
        float[] doc1Distr = agg.get();

        final Map<String, Float> doc2 = new HashMap<String, Float>();
        doc2.put("apples", 1.f);
        doc2.put("avocados", 1.f);
        doc2.put("colds", 1.f);
        doc2.put("flu", 1.f);
        doc2.put("like", 2.f);
        doc2.put("oranges", 1.f);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            evaluator.iterate(agg, new Object[] {word, doc2.get(word), labels[i], lambdas[i]});
        }
        float[] doc2Distr = agg.get();

        Assert.assertTrue(doc1Distr[0] > doc2Distr[0]);
        Assert.assertTrue(doc1Distr[1] < doc2Distr[1]);
    }


    @Test
    public void testMerge() throws Exception {
        udaf = new LDAPredictUDAF();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.STRING),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.INT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topics 2")};

        evaluator = udaf.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        agg = (LDAPredictUDAF.OnlineLDAPredictAggregationBuffer) evaluator.getNewAggregationBuffer();

        final Map<String, Float> doc = new HashMap<String, Float>();
        doc.put("apples", 1.f);
        doc.put("avocados", 1.f);
        doc.put("colds", 1.f);
        doc.put("flu", 1.f);
        doc.put("like", 2.f);
        doc.put("oranges", 1.f);

        Object[] partials = new Object[3];

        // bin #1
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < 6; i++) {
            evaluator.iterate(agg,
                new Object[] {words[i], doc.get(words[i]), labels[i], lambdas[i]});
        }
        partials[0] = evaluator.terminatePartial(agg);

        // bin #2
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 6; i < 12; i++) {
            evaluator.iterate(agg,
                new Object[] {words[i], doc.get(words[i]), labels[i], lambdas[i]});
        }
        partials[1] = evaluator.terminatePartial(agg);

        // bin #3
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 12; i < 18; i++) {
            evaluator.iterate(agg,
                new Object[] {words[i], doc.get(words[i]), labels[i], lambdas[i]});
        }

        partials[2] = evaluator.terminatePartial(agg);

        // merge in a different order
        final int[][] orders = new int[][] {{0, 1, 2}, {1, 0, 2}, {1, 2, 0}, {2, 1, 0}};
        for (int i = 0; i < orders.length; i++) {
            evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL2, partialOI);
            evaluator.reset(agg);

            evaluator.merge(agg, partials[orders[i][0]]);
            evaluator.merge(agg, partials[orders[i][1]]);
            evaluator.merge(agg, partials[orders[i][2]]);

            float[] distr = agg.get();
            Assert.assertTrue(distr[0] < distr[1]);
        }
    }

    @Test
    public void testUnmatchedTopics() throws Exception {
        udaf = new LDAPredictUDAF();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.STRING),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.INT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT)};

        evaluator = udaf.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        agg = (LDAPredictUDAF.OnlineLDAPredictAggregationBuffer) evaluator.getNewAggregationBuffer();

        final Map<String, Float> doc1 = new HashMap<String, Float>();
        doc1.put("fruits", 1.f);
        doc1.put("healthy", 1.f);
        doc1.put("vegetables", 1.f);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            evaluator.iterate(agg, new Object[] {word, doc1.get(word), labels[i], lambdas[i]});
        }
        float[] doc1Distr = agg.get();

        final Map<String, Float> doc2 = new HashMap<String, Float>();
        doc2.put("apples", 1.f);
        doc2.put("avocados", 1.f);
        doc2.put("colds", 1.f);
        doc2.put("flu", 1.f);
        doc2.put("like", 2.f);
        doc2.put("oranges", 1.f);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            evaluator.iterate(agg, new Object[] {word, doc2.get(word), labels[i], lambdas[i]});
        }
        float[] doc2Distr = agg.get();

        Assert.assertEquals(LDAUDTF.DEFAULT_TOPICS, doc1Distr.length);
        Assert.assertEquals(1.d, MathUtils.sum(doc1Distr), 1E-5d);

        Assert.assertEquals(LDAUDTF.DEFAULT_TOPICS, doc2Distr.length);
        Assert.assertEquals(1.d, MathUtils.sum(doc2Distr), 1E-5d);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testTerminateWithSameTopicProbability() throws Exception {
        udaf = new LDAPredictUDAF();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.STRING),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.INT),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                    PrimitiveObjectInspector.PrimitiveCategory.FLOAT),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-topics 2")};

        evaluator = udaf.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        agg = (LDAPredictUDAF.OnlineLDAPredictAggregationBuffer) evaluator.getNewAggregationBuffer();

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        // Assume that all words in a document are NOT in vocabulary that composes a LDA model.
        // Hence, the document should be assigned to topic #1 (#2) with probability 0.5 (0.5).
        for (int i = 0; i < 18; i++) {
            evaluator.iterate(agg, new Object[] {words[i], 0.f, labels[i], lambdas[i]});
        }

        // Probability for each of the two topics should be same.
        List<Object[]> result = (List<Object[]>) evaluator.terminate(agg);
        Assert.assertEquals(result.size(), 2);
        Assert.assertEquals(result.get(0)[1], result.get(1)[1]);
    }

}
