package hivemall.evaluation;


import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.SimpleGenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
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
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector
        };

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
    public void test() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(0.5d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});
        Assert.assertEquals(0.625, agg.get(), 1e-4);
    }

    @Test(expected = HiveException.class)
    public void testZeroBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        // FMeasure for beta has zero value is not defined
        agg.get();
    }

    @Test(expected = HiveException.class)
    public void testNegativeBeta() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(-1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        // FMeasure for beta has negative value is not defined
        agg.get();
    }

    @Test
    public void testF1score() throws Exception {
        List<Integer> actual = Arrays.asList(1, 3, 2, 6);
        List<Integer> predicted = Arrays.asList(1, 2, 4);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});
        Assert.assertEquals(0.5714285714285715d, agg.get(), 1e-5);
    }

    @Test
    public void testMaxFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(1, 2, 3);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(1.d, agg.get(), 1e-5);
    }

    @Test
    public void testMinFMeasure() throws Exception {
        List<Integer> actual = Arrays.asList(0, 0, 0);
        List<Integer> predicted = Arrays.asList(1, 2, 3);
        DoubleWritable beta = new DoubleWritable(1.0d);

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        evaluator.iterate(agg, new Object[] {actual, predicted, beta});

        Assert.assertEquals(-1.d, agg.get(), 1e-5);
    }



}