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
package hivemall.fm;

import hivemall.TestUtils;
import hivemall.utils.lang.NumberUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class FieldAwareFactorizationMachineUDTFTest {

    private static final boolean DEBUG = false;
    private static final int ITERATIONS = 50;
    private static final int MAX_LINES = 200;

    // ----------------------------------------------------
    // bigdata.tr.txt

    @Test
    public void testSGD() throws HiveException, IOException {
        runIterations("Pure SGD test", "bigdata.tr.txt.gz",
            "-opt sgd -classification -factors 10 -w0 -seed 43", 0.60f);
    }

    @Test
    public void testAdaGrad() throws HiveException, IOException {
        runIterations("AdaGrad test", "bigdata.tr.txt.gz",
            "-opt adagrad -classification -factors 10 -w0 -seed 43", 0.30f);
    }

    @Test
    public void testAdaGradNoCoeff() throws HiveException, IOException {
        runIterations("AdaGrad No Coeff test", "bigdata.tr.txt.gz",
            "-opt adagrad -no_coeff -classification -factors 10 -w0 -seed 43", 0.30f);
    }

    @Test
    public void testFTRL() throws HiveException, IOException {
        runIterations("FTRL test", "bigdata.tr.txt.gz",
            "-opt ftrl -classification -factors 10 -w0 -seed 43", 0.30f);
    }

    @Test
    public void testFTRLNoCoeff() throws HiveException, IOException {
        runIterations("FTRL Coeff test", "bigdata.tr.txt.gz",
            "-opt ftrl -no_coeff -classification -factors 10 -w0 -seed 43", 0.30f);
    }

    // ----------------------------------------------------
    // https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz

    @Test
    public void testSample() throws IOException, HiveException {
        System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2");
        run("[Sample.ffm] default option",
            "https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz",
            "-classification -factors 2 -iters 10 -feature_hashing 20 -seed 43", 0.01f);
    }

    // TODO @Test
    public void testSampleEnableNorm() throws IOException, HiveException {
        System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2");
        run("[Sample.ffm] default option",
            "https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz",
            "-classification -factors 2 -iters 10 -feature_hashing 20 -seed 43 -enable_norm",
            0.01f);
    }

    private static void run(String testName, String testFile, String testOptions,
            float lossThreshold) throws IOException, HiveException {
        println(testName);

        FieldAwareFactorizationMachineUDTF udtf = new FieldAwareFactorizationMachineUDTF();
        ObjectInspector[] argOIs =
                new ObjectInspector[] {
                        ObjectInspectorFactory.getStandardListObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            testOptions)};

        udtf.initialize(argOIs);
        FieldAwareFactorizationMachineModel model = udtf.initModel(udtf._params);
        Assert.assertTrue("Actual class: " + model.getClass().getName(),
            model instanceof FFMStringFeatureMapModel);

        int lines = 0;
        BufferedReader data = readFile(testFile);
        while (true) {
            //gather features in current line
            final String input = data.readLine();
            if (input == null) {
                break;
            }
            lines++;
            String[] featureStrings = input.split(" ");

            double y = Double.parseDouble(featureStrings[0]);
            if (y == 0) {
                y = -1;//LibFFM data uses {0, 1}; Hivemall uses {-1, 1}
            }

            final List<String> features = new ArrayList<String>(featureStrings.length - 1);
            for (int j = 1; j < featureStrings.length; ++j) {
                String fj = featureStrings[j];
                String[] splitted = fj.split(":");
                Assert.assertEquals(3, splitted.length);
                String indexStr = splitted[1];
                String f = fj;
                if (NumberUtils.isDigits(indexStr)) {
                    int index = Integer.parseInt(indexStr) + 1; // avoid 0 index
                    f = splitted[0] + ':' + index + ':' + splitted[2];
                }
                features.add(f);
            }

            udtf.process(new Object[] {features, y});
        }
        udtf.finalizeTraining();
        data.close();

        println("model size=" + udtf._model.getSize());

        double avgLoss = udtf._cvState.getAverageLoss(lines);
        Assert.assertTrue("Last loss was greater than expected: " + avgLoss,
            avgLoss < lossThreshold);
    }

    private static void runIterations(String testName, String testFile, String testOptions,
            float lossThreshold) throws IOException, HiveException {
        println(testName);

        FieldAwareFactorizationMachineUDTF udtf = new FieldAwareFactorizationMachineUDTF();
        ObjectInspector[] argOIs =
                new ObjectInspector[] {
                        ObjectInspectorFactory.getStandardListObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            testOptions)};

        udtf.initialize(argOIs);
        FieldAwareFactorizationMachineModel model = udtf.initModel(udtf._params);
        Assert.assertTrue("Actual class: " + model.getClass().getName(),
            model instanceof FFMStringFeatureMapModel);

        double loss = 0.d;
        double cumul = 0.d;
        for (int trainingIteration = 1; trainingIteration <= ITERATIONS; ++trainingIteration) {
            BufferedReader data = readFile(testFile);
            loss = udtf._cvState.getCumulativeLoss();
            int lines = 0;
            for (int lineNumber = 0; lineNumber < MAX_LINES; ++lineNumber, ++lines) {
                //gather features in current line
                final String input = data.readLine();
                if (input == null) {
                    break;
                }
                String[] featureStrings = input.split(" ");

                double y = Double.parseDouble(featureStrings[0]);
                if (y == 0) {
                    y = -1;//LibFFM data uses {0, 1}; Hivemall uses {-1, 1}
                }

                final List<String> features = new ArrayList<String>(featureStrings.length - 1);
                for (int j = 1; j < featureStrings.length; ++j) {
                    String fj = featureStrings[j];
                    String[] splitted = fj.split(":");
                    Assert.assertEquals(3, splitted.length);
                    String indexStr = splitted[1];
                    String f = fj;
                    if (NumberUtils.isDigits(indexStr)) {
                        int index = Integer.parseInt(indexStr) + 1; // avoid 0 index
                        f = splitted[0] + ':' + index + ':' + splitted[2];
                    }
                    features.add(f);
                }
                udtf.process(new Object[] {features, y});
            }
            cumul = udtf._cvState.getCumulativeLoss();
            loss = (cumul - loss) / lines;
            println(trainingIteration + " " + loss + " " + cumul / (trainingIteration * lines));
            data.close();
        }
        println("model size=" + udtf._model.getSize());
        Assert.assertTrue("Last loss was greater than expected: " + loss, loss < lossThreshold);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(FieldAwareFactorizationMachineUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-opt sgd -classification -factors 10 -w0 -seed 43")},
            new Object[][] {{Arrays.asList("0:1:-2", "1:2:-1"), 1.0}});
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is;
        if (fileName.startsWith("http")) {
            URL url = new URL(fileName);
            is = url.openStream();
        } else {
            is = FieldAwareFactorizationMachineUDTFTest.class.getResourceAsStream(fileName);
        }
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    private static void println(String line) {
        if (DEBUG) {
            System.out.println(line);
        }
    }

}
