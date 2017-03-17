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
package hivemall.evaluation;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.SimpleGenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import java.util.ArrayList;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class AUCUDAFTest {
    AUCUDAF auc;
    GenericUDAFEvaluator evaluator;
    ObjectInspector[] inputOIs;
    ObjectInspector[] partialOI;
    AUCUDAF.ClassificationAUCAggregationBuffer agg;

    @Before
    public void setUp() throws Exception {
        auc = new AUCUDAF();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                        PrimitiveObjectInspector.PrimitiveCategory.DOUBLE),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                        PrimitiveObjectInspector.PrimitiveCategory.INT)};

        evaluator = auc.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("a");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("minScore");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("fp");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("tp");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("fpPrev");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("tpPrev");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);

        MapObjectInspector partialAreasOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("partialAreas");
        fieldOIs.add(partialAreasOI);

        MapObjectInspector fpCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
            PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("fpCounts");
        fieldOIs.add(fpCountsOI);

        MapObjectInspector tpCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
            PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("tpCounts");
        fieldOIs.add(tpCountsOI);

        partialOI = new ObjectInspector[2];
        partialOI[0] = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);

        agg = (AUCUDAF.ClassificationAUCAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    @Test
    public void test() throws Exception {
        // should be sorted by scores in a descending order
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {1, 1, 0, 1, 0};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels[i]});
        }

        Assert.assertEquals(0.83333, agg.get(), 1e-5);
    }

    @Test(expected=HiveException.class)
    public void testAllTruePositive() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {1, 1, 1, 1, 1};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels[i]});
        }

        // AUC for all TP scores are not defined
        agg.get();
    }

    @Test(expected=HiveException.class)
    public void testAllFalsePositive() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {0, 0, 0, 0, 0};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels[i]});
        }

        // AUC for all FP scores are not defined
        agg.get();
    }

    @Test
    public void testMaxAUC() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {1, 1, 1, 1, 0};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels[i]});
        }

        // All TPs are ranked higher than FPs => AUC=1.0
        Assert.assertEquals(1.d, agg.get(), 1e-5);
    }

    @Test
    public void testMinAUC() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {0, 0, 0, 1, 1};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels[i]});
        }

        // All TPs are ranked lower than FPs => AUC=0.0
        Assert.assertEquals(0.d, agg.get(), 1e-5);
    }

    @Test
    public void testMidAUC() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};

        // if TP and FP appear alternately, AUC=0.5
        final int[] labels1 = new int[] {1, 0, 1, 0, 1};
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels1[i]});
        }
        Assert.assertEquals(0.5, agg.get(), 1e-5);

        final int[] labels2 = new int[] {0, 1, 0, 1, 0};
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        for (int i = 0; i < scores.length; i++) {
            evaluator.iterate(agg, new Object[] {scores[i], labels2[i]});
        }
        Assert.assertEquals(0.5, agg.get(), 1e-5);
    }

    @Test
    public void testMerge() throws Exception {
        final double[] scores = new double[] {0.8, 0.7, 0.5, 0.3, 0.2};
        final int[] labels = new int[] {1, 1, 0, 1, 0};

        Object[] partials = new Object[3];

        // bin #1
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        evaluator.iterate(agg, new Object[] {scores[0], labels[0]});
        evaluator.iterate(agg, new Object[] {scores[1], labels[1]});
        partials[0] = evaluator.terminatePartial(agg);

        // bin #2
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        evaluator.iterate(agg, new Object[] {scores[2], labels[2]});
        evaluator.iterate(agg, new Object[] {scores[3], labels[3]});
        partials[1] = evaluator.terminatePartial(agg);

        // bin #3
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);
        evaluator.iterate(agg, new Object[] {scores[4], labels[4]});
        partials[2] = evaluator.terminatePartial(agg);

        // merge bins
        // merge in a different order; e.g., <bin0, bin1>, <bin1, bin0> should return same value
        final int[][] orders = new int[][] {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 1, 0}, {2, 0, 1}};
        for (int i = 0; i < orders.length; i++) {
            evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL2, partialOI);
            evaluator.reset(agg);

            evaluator.merge(agg, partials[orders[i][0]]);
            evaluator.merge(agg, partials[orders[i][1]]);
            evaluator.merge(agg, partials[orders[i][2]]);

            Assert.assertEquals(0.83333, agg.get(), 1e-5);
        }
    }
}
