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
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;


public class FMeasureUDAFTest {
    FMeasureUDAF fmeasure;
    GenericUDAFEvaluator evaluator;
    ObjectInspector[] inputOIs;
    FMeasureUDAF.FMeasureAggregationBuffer agg;

    @Before
    public void setUp() throws Exception {
        fmeasure = new FMeasureUDAF();
        inputOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-beta 1.")};

        evaluator = fmeasure.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));

        agg = (FMeasureUDAF.FMeasureAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    private void setUpWithArguments(double beta, String average) throws Exception {
        fmeasure = new FMeasureUDAF();
        inputOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-beta " + beta
                            + " -average " + average)};

        evaluator = fmeasure.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));
        agg = (FMeasureUDAF.FMeasureAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    private void binarySetUp(Object actual, Object predicted, double beta, String average)
            throws Exception {
        fmeasure = new FMeasureUDAF();
        inputOIs = new ObjectInspector[3];

        String actualClassName = actual.getClass().getName();
        if (actualClassName.equals("java.lang.Integer")) {
            inputOIs[0] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.INT);
        } else if (actualClassName.equals("java.lang.Boolean")) {
            inputOIs[0] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.BOOLEAN);
        } else if ((actualClassName.equals("java.lang.String"))) {
            inputOIs[0] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.STRING);
        }

        String predicatedClassName = predicted.getClass().getName();
        if (predicatedClassName.equals("java.lang.Integer")) {
            inputOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.INT);
        } else if (predicatedClassName.equals("java.lang.Boolean")) {
            inputOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.BOOLEAN);
        } else if ((predicatedClassName.equals("java.lang.String"))) {
            inputOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.STRING);
        }

        inputOIs[2] = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-beta " + beta
                    + " -average " + average);

        evaluator = fmeasure.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));
        agg = (FMeasureUDAF.FMeasureAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    @Test
    public void testBinaryMultiSamplesAverageBinary() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        double beta = 1.;
        String average = "binary";
        binarySetUp(actual[0], predicted[0], beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        // should equal to turi's result
        // https://turi.com/learn/userguide/evaluation/classification.html#fscores-f1-fbeta-
        Assert.assertEquals(0.3333d, agg.get(), 1e-4);
    }

    @Test(expected = HiveException.class)
    public void testBinaryMultiSamplesAverageMacro() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        double beta = 1.;
        String average = "macro";
        binarySetUp(actual[0], predicted[0], beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        agg.get();
    }

    @Test
    public void testBinaryMultiSamples() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        double beta = 1.;
        String average = "micro";
        binarySetUp(actual[0], predicted[0], beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        Assert.assertEquals(0.5d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryMultiSamplesBeta2() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        double beta = 2.0;
        String average = "binary";
        binarySetUp(actual[0], predicted[0], beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        Assert.assertEquals(0.4166d, agg.get(), 1e-4);
    }

    @Test
    public void testBinary() throws Exception {
        int actual = 1;
        int predicted = 1;
        double beta = 1.0;
        String average = "micro";
        binarySetUp(actual, predicted, beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(1.d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryNegativeInput() throws Exception {
        int actual = 1;
        int predicted = -1;
        binarySetUp(actual, predicted, 1.0, "binary");

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(0.d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryBooleanInput() throws Exception {
        boolean actual = true;
        boolean predicted = false;
        double beta = 1.0d;
        binarySetUp(actual, predicted, beta, "binary");

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(0.d, agg.get(), 1e-4);
    }

    @Test(expected = HiveException.class)
    public void testBinaryInvalidStringInput() throws Exception {
        String actual = "cat";
        int predicted = 1;
        binarySetUp(actual, predicted, 1.0, "micro");

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        agg.get();
    }

    @Test(expected = HiveException.class)
    public void testBinaryInvalidLargeIntInput() throws Exception {
        int actual = 1;
        int predicted = 3;
        binarySetUp(actual, predicted, 1.0, "micro");

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        agg.get();
    }

    @Test(expected = HiveException.class)
    public void testMultiLabelZeroBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        double beta = 0.;
        setUpWithArguments(beta, "micro");

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        // FMeasure for beta has zero value is not defined
        agg.get();
    }

    @Test(expected = HiveException.class)
    public void testMultiLabelNegativeBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        double beta = -1.0d;
        String average = "micro";
        setUpWithArguments(beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        // FMeasure for beta has negative value is not defined
        agg.get();
    }

    @Test
    public void testMultiLabelF1score() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        double beta = 1.0;
        String average = " micro";
        setUpWithArguments(beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        // should equal to spark's micro f1 measure result
        // https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#multilabel-classification
        Assert.assertEquals(0.5714285714285714, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelMaxFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(1, 2, 3);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        double beta = 1.0;
        String average = "micro";
        setUpWithArguments(beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(1.d, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelMinFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(0, 0, 0);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        double beta = 1.0;
        String average = "micro";
        setUpWithArguments(beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(0.d, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelF1MultiSamples() throws Exception {
        String[][] actual = { {"0", "2"}, {"0", "1"}, {"0"}, {"2"}, {"2", "0"}, {"0", "1"},
                {"1", "2"}};
        String[][] predicted = { {"0", "1"}, {"0", "2"}, {}, {"2"}, {"2", "0"}, {"0", "1", "2"},
                {"1"}};

        double beta = 1.0;
        String average = "micro";
        setUpWithArguments(beta, average);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        // should equal to spark's micro f1 measure result
        // https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#multilabel-classification
        Assert.assertEquals(0.6956d, agg.get(), 1e-4);
    }

    @Test
    public void testMultiLabelFmeasureMultiSamples() throws Exception {
        String[][] actual = { {"0", "2"}, {"0", "1"}, {"0"}, {"2"}, {"2", "0"}, {"0", "1"},
                {"1", "2"}};
        String[][] predicted = { {"0", "1"}, {"0", "2"}, {}, {"2"}, {"2", "0"}, {"0", "1", "2"},
                {"1"}};

        double beta = 2.0;
        String average = "micro";
        setUpWithArguments(beta, average);
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        Assert.assertEquals(0.6779d, agg.get(), 1e-4);
    }

    @Test(expected = HiveException.class)
    public void testMultiLabelFmeasureBinary() throws Exception {
        String[][] actual = { {"0", "2"}, {"0", "1"}, {"0"}, {"2"}, {"2", "0"}, {"0", "1"},
                {"1", "2"}};
        String[][] predicted = { {"0", "1"}, {"0", "2"}, {}, {"2"}, {"2", "0"}, {"0", "1", "2"},
                {"1"}};

        double beta = 1.0;
        String average = "binary";

        setUpWithArguments(beta, average);
        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i]});
        }

        agg.get();
    }
}
