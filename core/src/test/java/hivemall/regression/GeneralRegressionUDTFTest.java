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
package hivemall.regression;

import static hivemall.utils.hadoop.HiveUtils.lazyInteger;
import static hivemall.utils.hadoop.HiveUtils.lazyLong;
import static hivemall.utils.hadoop.HiveUtils.lazyString;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.lazy.LazyInteger;
import org.apache.hadoop.hive.serde2.lazy.LazyLong;
import org.apache.hadoop.hive.serde2.lazy.LazyString;
import org.apache.hadoop.hive.serde2.lazy.objectinspector.primitive.LazyPrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.lazy.objectinspector.primitive.LazyStringObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class GeneralRegressionUDTFTest {
    private static final boolean DEBUG = false;

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedOptimizer() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-opt UnsupportedOpt");

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedLossFunction() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss UnsupportedLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidLossFunction() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss HingeLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedRegularization() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reg UnsupportedReg");

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});
    }

    @Test
    public void testNoOptions() throws Exception {
        List<String> x = Arrays.asList("1:-2", "2:-1");
        float y = 0.f;

        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI});

        udtf.process(new Object[] {x, y});

        udtf.finalizeTraining();

        float predicted = udtf.predict(udtf.parseFeatures(x));
        Assert.assertEquals(y, predicted, 1E-5);
    }

    private <T> void testFeature(@Nonnull List<T> x, @Nonnull ObjectInspector featureOI,
            @Nonnull Class<T> featureClass, @Nonnull Class<?> modelFeatureClass) throws Exception {
        float y = 0.f;

        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector valueOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ListObjectInspector featureListOI = ObjectInspectorFactory.getStandardListObjectInspector(featureOI);

        udtf.initialize(new ObjectInspector[] {featureListOI, valueOI});

        final List<Object> modelFeatures = new ArrayList<Object>();
        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                Object[] forwardMapObj = (Object[]) input;
                modelFeatures.add(forwardMapObj[0]);
            }
        });

        udtf.process(new Object[] {x, y});

        udtf.close();

        Assert.assertFalse(modelFeatures.isEmpty());
        for (Object modelFeature : modelFeatures) {
            Assert.assertEquals("All model features must have same type", modelFeatureClass,
                modelFeature.getClass());
        }
    }

    @Test
    public void testLazyStringFeature() throws Exception {
        LazyStringObjectInspector oi = LazyPrimitiveObjectInspectorFactory.getLazyStringObjectInspector(
            false, (byte) 0);
        List<LazyString> x = Arrays.asList(lazyString("テスト:-2", oi), lazyString("漢字:-333.0", oi),
            lazyString("test:-1"));
        testFeature(x, oi, LazyString.class, String.class);
    }

    @Test
    public void testStringFeature() throws Exception {
        List<String> x = Arrays.asList("1:-2", "2:-1");
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        testFeature(x, featureOI, String.class, String.class);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testIlleagalStringFeature() throws Exception {
        List<String> x = Arrays.asList("1:-2jjjj", "2:-1");
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        testFeature(x, featureOI, String.class, String.class);
    }

    @Test
    public void testTextFeature() throws Exception {
        List<Text> x = Arrays.asList(new Text("1:-2"), new Text("2:-1"));
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        testFeature(x, featureOI, Text.class, String.class);
    }

    @Test
    public void testIntegerFeature() throws Exception {
        List<Integer> x = Arrays.asList(111, 222);
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        testFeature(x, featureOI, Integer.class, Integer.class);
    }

    @Test
    public void testLazyIntegerFeature() throws Exception {
        List<LazyInteger> x = Arrays.asList(lazyInteger(111), lazyInteger(222));
        ObjectInspector featureOI = LazyPrimitiveObjectInspectorFactory.LAZY_INT_OBJECT_INSPECTOR;
        testFeature(x, featureOI, LazyInteger.class, Integer.class);
    }

    @Test
    public void testWritableIntFeature() throws Exception {
        List<IntWritable> x = Arrays.asList(new IntWritable(111), new IntWritable(222));
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.writableIntObjectInspector;
        testFeature(x, featureOI, IntWritable.class, Integer.class);
    }

    @Test
    public void testLongFeature() throws Exception {
        List<Long> x = Arrays.asList(111L, 222L);
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.javaLongObjectInspector;
        testFeature(x, featureOI, Long.class, Long.class);
    }

    @Test
    public void testLazyLongFeature() throws Exception {
        List<LazyLong> x = Arrays.asList(lazyLong(111), lazyLong(222));
        ObjectInspector featureOI = LazyPrimitiveObjectInspectorFactory.LAZY_LONG_OBJECT_INSPECTOR;
        testFeature(x, featureOI, LazyLong.class, Long.class);
    }

    @Test
    public void testWritableLongFeature() throws Exception {
        List<LongWritable> x = Arrays.asList(new LongWritable(111L), new LongWritable(222L));
        ObjectInspector featureOI = PrimitiveObjectInspectorFactory.writableLongObjectInspector;
        testFeature(x, featureOI, LongWritable.class, Long.class);
    }

    private void run(@Nonnull String options) throws Exception {
        println(options);

        int numSamples = 100;

        float x1Min = -5.f, x1Max = 5.f;
        float x1Step = (x1Max - x1Min) / numSamples;

        float x2Min = -3.f, x2Max = 3.f;
        float x2Step = (x2Max - x2Min) / numSamples;

        ArrayList<List<String>> samplesList = new ArrayList<List<String>>(numSamples);
        ArrayList<Float> ys = new ArrayList<Float>(numSamples);
        float x1 = x1Min, x2 = x2Min;

        for (int i = 0; i < numSamples; i++) {
            samplesList.add(Arrays.asList("1:" + String.valueOf(x1), "2:" + String.valueOf(x2)));

            ys.add(x1 * 0.5f);

            x1 += x1Step;
            x2 += x2Step;
        }

        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, options);

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});

        float accum = 0.f;
        for (int i = 0; i < numSamples; i++) {
            float y = ys.get(i).floatValue();
            float predicted = udtf.predict(udtf.parseFeatures(samplesList.get(i)));
            accum += Math.abs(y - predicted);
        }
        float maeInit = accum / numSamples;
        println("Mean absolute error before training: " + maeInit);

        for (int i = 0; i < numSamples; i++) {
            udtf.process(new Object[] {samplesList.get(i), (Float) ys.get(i)});
        }

        udtf.finalizeTraining();

        double cumLoss = udtf.getCumulativeLoss();
        println("Cumulative loss: " + cumLoss);
        double normalizedLoss = cumLoss / numSamples;
        Assert.assertTrue("cumLoss: " + cumLoss + ", normalizedLoss: " + normalizedLoss
                + "\noptions: " + options, normalizedLoss < 0.1d);

        accum = 0.f;
        for (int i = 0; i < numSamples; i++) {
            float y = ys.get(i).floatValue();

            float predicted = udtf.predict(udtf.parseFeatures(samplesList.get(i)));
            println("Predicted: " + predicted + ", Actual: " + y);

            accum += Math.abs(y - predicted);
        }
        float mae = accum / numSamples;
        println("Mean absolute error after training: " + mae);
        Assert.assertTrue("accum: " + accum + ", mae (init):" + maeInit + ", mae:" + mae
                + "\noptions: " + options, mae < maeInit);
    }

    @Test
    public void test() throws Exception {
        String[] optimizers = new String[] {"SGD", "AdaDelta", "AdaGrad", "Adam"};
        String[] regularizations = new String[] {"NO", "L1", "L2", "ElasticNet", "RDA"};
        String[] lossFunctions = new String[] {"SquaredLoss", "QuantileLoss",
                "EpsilonInsensitiveLoss", "SquaredEpsilonInsensitiveLoss", "HuberLoss"};

        for (String opt : optimizers) {
            for (String reg : regularizations) {
                if (reg == "RDA" && opt != "AdaGrad") {
                    continue;
                }

                for (String loss : lossFunctions) {
                    String options = "-opt " + opt + " -reg " + reg + " -loss " + loss
                            + " -iter 512";

                    // sparse
                    run(options);

                    // mini-batch
                    if (opt != "AdaGrad") {
                        options += " -mini_batch 10";
                        run(options);
                    }

                    // dense
                    options += " -dense";
                    run(options);
                }
            }
        }
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
