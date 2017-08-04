package hivemall.evaluation;


import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.SimpleGenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class FMeasureUDAFTest {
    FMeasureUDAF fmeasure;
    GenericUDAFEvaluator evaluator;
    ObjectInspector[] inputOIs;
    ObjectInspector[] partialOI;
    FMeasureUDAF.FMeasureAggregationBuffer agg;

    @Before
    public void setUp() throws Exception {
        fmeasure = new FMeasureUDAF();
        inputOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector};

        evaluator = fmeasure.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));
        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("tp");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("totalActual");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("totalPredicted");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("beta");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        partialOI = new ObjectInspector[2];
        partialOI[0] = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);

        agg = (FMeasureUDAF.FMeasureAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    private void binarySetUp(Object actual, Object predicted) throws Exception {
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

        inputOIs[2] = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
        evaluator = fmeasure.getEvaluator(new SimpleGenericUDAFParameterInfo(inputOIs, false, false));
        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("tp");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("totalActual");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("totalPredicted");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        fieldNames.add("beta");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        partialOI = new ObjectInspector[2];
        partialOI[0] = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);

        agg = (FMeasureUDAF.FMeasureAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    @Test
    public void testBinaryMultiSamples() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        binarySetUp(actual[0], predicted[0]);
        DoubleWritable beta = new DoubleWritable(1.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i], beta});
        }

        // should equal to turi's result
        // https://turi.com/learn/userguide/evaluation/classification.html#fscores-f1-fbeta-
        Assert.assertEquals(0.3333d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryMultiSamplesBeta2() throws Exception {
        final int[] actual = {0, 1, 0, 0, 0, 1, 0, 0};
        final int[] predicted = {1, 0, 0, 1, 0, 1, 0, 1};
        binarySetUp(actual[0], predicted[0]);
        DoubleWritable beta = new DoubleWritable(2.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i], beta});
        }

        // should equal to turi's result
        // https://turi.com/learn/userguide/evaluation/classification.html#fscores-f1-fbeta-
        Assert.assertEquals(0.4166d, agg.get(), 1e-4);
    }

    @Test
    public void testBinary() throws Exception {
        int actual = 1;
        int predicted = 1;
        binarySetUp(actual, predicted);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted});

        Assert.assertEquals(1.d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryNegativeInput() throws Exception {
        int actual = 1;
        int predicted = -1;
        binarySetUp(actual, predicted);

        DoubleWritable beta = new DoubleWritable(1.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(-1.d, agg.get(), 1e-4);
    }

    @Test
    public void testBinaryBooleanInput() throws Exception {
        boolean actual = true;
        boolean predicted = false;
        DoubleWritable beta = new DoubleWritable(1.d);
        binarySetUp(actual, predicted);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(-1.d, agg.get(), 1e-4);
    }

    @Test(expected = HiveException.class)
    public void testBinaryInvalidStringInput() throws Exception {
        String actual = "cat";
        int predicted = 1;
        binarySetUp(actual, predicted);
        DoubleWritable beta = new DoubleWritable(1.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        agg.get();
    }


    @Test(expected = HiveException.class)
    public void testBinaryInvalidLargeIntInput() throws Exception {
        int actual = 1;
        int predicted = 3;
        binarySetUp(actual, predicted);
        DoubleWritable beta = new DoubleWritable(1.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        agg.get();
    }


    @Test(expected = HiveException.class)
    public void testMultiLabelZeroBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(0.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        // FMeasure for beta has zero value is not defined
        agg.get();
    }

    @Test(expected = HiveException.class)
    public void testMultiLabelNegativeBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(-1.d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        // FMeasure for beta has negative value is not defined
        agg.get();
    }

    @Test
    public void testMultiLabelF1score() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(0.5714285714285715d, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(0.5d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(0.625, agg.get(), 1e-4);
    }

    @Test
    public void testMultiLabelMaxFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(1, 2, 3);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(1.d, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelMinFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(0, 0, 0);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(0.d, agg.get(), 1e-5);
    }

    @Test
    public void testMultiLabelF1MultiSamples() throws Exception {
        String[][] actual = { {"0", "1"}, {"0", "2"}, {}, {"2"}, {"2", "0"}, {"0", "1", "2"}, {"1"}};

        String[][] predicted = { {"0", "2"}, {"0", "1"}, {"0"}, {"2"}, {"2", "0"}, {"0", "1"},
                {"1", "2"}};

        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i], beta});
        }

        // should equal to spark's micro f1 measure result
        // https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#multilabel-classification
        Assert.assertEquals(0.6956d, agg.get(), 1e-4);
    }

    @Test
    public void testMultiLabelFmeasureMultiSamples() throws Exception {
        String[][] actual = { {"0", "1"}, {"0", "2"}, {}, {"2"}, {"2", "0"}, {"0", "1", "2"}, {"1"}};

        String[][] predicted = { {"0", "2"}, {"0", "1"}, {"0"}, {"2"}, {"2", "0"}, {"0", "1"},
                {"1", "2"}};

        DoubleWritable beta = new DoubleWritable(2.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < actual.length; i++) {
            evaluator.iterate(agg, new Object[] {actual[i], predicted[i], beta});
        }

        Assert.assertEquals(0.6779d, agg.get(), 1e-4);
    }
}
