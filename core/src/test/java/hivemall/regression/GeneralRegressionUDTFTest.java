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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Assert;
import org.junit.Test;

import javax.annotation.Nonnull;

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

    private void run(@Nonnull String options) throws Exception {
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

        int numTrain = (int) (numSamples * 0.8);
        int maxIter = 256;

        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, options);

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});

        float lossAvgPrev = Float.MAX_VALUE;
        float lossAvg = 0.f;
        int it = 0;
        while ((it < maxIter) && (Math.abs(lossAvg - lossAvgPrev) > 1e-3f)) {
            lossAvgPrev = lossAvg;
            lossAvg = 0.f;
            for (int i = 0; i < numTrain; i++) {
                udtf.process(new Object[] {samplesList.get(i), (Float) ys.get(i)});
                lossAvg += udtf.getLoss();
            }
            lossAvg /= numTrain;
            println("Iter: " + ++it + ", Avg. loss: " + lossAvg);
        }
        Assert.assertTrue(lossAvg < 0.1f);

        float accum = 0.f;

        for (int i = numTrain; i < numSamples; i++) {
            float y = ys.get(i).floatValue();

            float predicted = udtf.predict(udtf.parseFeatures(samplesList.get(i)));
            println("Predicted: " + predicted + ", Actual: " + y);

            accum += Math.abs(y - predicted);
        }

        float err = accum / (numSamples - numTrain);
        println("Mean absolute error: " + err);
        Assert.assertTrue(err < 0.2f);
    }

    @Test
    public void test() throws Exception {
        String[] optimizers = new String[] {"SGD", "AdaDelta", "AdaGrad", "Adam"};
        String[] regularizations = new String[] {"NO", "L1", "L2", "ElasticNet", "RDA"};
        String[] lossFunctions = new String[] {"SquaredLoss", "QuantileLoss",
                "EpsilonInsensitiveLoss", "HuberLoss"};

        for (String opt : optimizers) {
            for (String reg : regularizations) {
                if (reg == "RDA" && opt != "AdaGrad") {
                    continue;
                }

                for (String loss : lossFunctions) {
                    String options = "-opt " + opt + " -reg " + reg + " -loss " + loss
                            + " -lambda 1e-6 -eta0 1e-1";
                    println(options);

                    // sparse
                    run(options);

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
