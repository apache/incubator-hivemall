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
package hivemall.factorization.fm;

import hivemall.TestUtils;
import hivemall.factorization.fm.FFMStringFeatureMapModel;
import hivemall.factorization.fm.FieldAwareFactorizationMachineModel;
import hivemall.factorization.fm.FieldAwareFactorizationMachineUDTF;
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

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class FieldAwareFactorizationMachineUDTFTest {

    private static final boolean DEBUG = false;

    // ----------------------------------------------------
    // bigdata.tr.txt

    @Test
    public void testSGD() throws HiveException, IOException {
        run("Pure SGD test", "bigdata.tr.txt.gz",
            "-opt sgd -linear_term -classification -factors 10 -w0 -eta 0.4 -iters 20 -seed 43",
            0.30f);
    }

    @Test
    public void testAdaGrad() throws HiveException, IOException {
        run("AdaGrad test", "bigdata.tr.txt.gz",
            "-opt adagrad -linear_term -classification -factors 10 -w0 -eta 0.4 -iters 30 -seed 43",
            0.30f);
    }

    @Test
    public void testAdaGradNoCoeff() throws HiveException, IOException {
        run("AdaGrad No Coeff test", "bigdata.tr.txt.gz",
            "-opt adagrad -classification -factors 10 -w0 -eta 0.4 -iters 30 -seed 43", 0.30f);
    }

    @Test
    public void testFTRL() throws HiveException, IOException {
        run("FTRL test", "bigdata.tr.txt.gz",
            "-opt ftrl -linear_term -classification -factors 10 -w0 -alphaFTRL 10.0 -seed 43",
            0.30f);
    }

    @Test
    public void testFTRLNoCoeff() throws HiveException, IOException {
        run("FTRL Coeff test", "bigdata.tr.txt.gz",
            "-opt ftrl -classification -factors 10 -w0 -alphaFTRL 10.0 -seed 43", 0.30f);
    }

    // ----------------------------------------------------
    // https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz

    @Test
    public void testSampleDisableNorm() throws IOException, HiveException {
        System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2");
        run("[Sample.ffm] default option",
            "https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz",
            "-disable_norm -linear_term -classification -factors 2 -feature_hashing 20 -seed 43",
            0.01f);
    }

    @Test
    public void testSample() throws IOException, HiveException {
        System.setProperty("https.protocols", "TLSv1,TLSv1.1,TLSv1.2");
        run("[Sample.ffm] default option",
            "https://github.com/myui/ml_dataset/raw/master/ffm/sample.ffm.gz",
            "-linear_term -classification -factors 2 -alphaFTRL 10.0 -feature_hashing 20 -seed 43",
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

    @Test
    public void testEarlyStopping() throws HiveException, IOException {
        println("Early stopping");

        int iters = 20;

        FieldAwareFactorizationMachineUDTF udtf = new FieldAwareFactorizationMachineUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-opt sgd -linear_term -classification -factors 10 -w0 -eta 0.4 -iters " + iters
                            + " -early_stopping -validation_threshold 1 -disable_cv -seed 43")};

        udtf.initialize(argOIs);

        BufferedReader data = readFile("bigdata.tr.txt.gz");
        List<List<String>> featureVectors = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        while (true) {
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
            ys.add(y);

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
            featureVectors.add(features);

            udtf.process(new Object[] {features, y});
        }
        udtf.finalizeTraining();
        data.close();

        double loss = udtf._validationState.getAverageLoss(featureVectors.size());
        Assert.assertTrue(
            "Training seems to be failed because average loss is greater than 0.6: " + loss,
            loss <= 0.6);

        Assert.assertNotNull("Early stopping validation has not been conducted",
            udtf._validationState);
        println("Performed " + udtf._validationState.getCurrentIteration() + " iterations out of "
                + iters);
        Assert.assertNotEquals("Early stopping did not happen", iters,
            udtf._validationState.getCurrentIteration());

        // store the best state achieved by early stopping
        iters = udtf._validationState.getCurrentIteration() - 2; // best loss was at (N-2)-th iter
        double cumulativeLoss = udtf._validationState.getCumulativeLoss();
        println("Cumulative loss: " + cumulativeLoss);

        // train with the number of early-stopped iterations
        udtf = new FieldAwareFactorizationMachineUDTF();
        argOIs[2] = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            "-opt sgd -linear_term -classification -factors 10 -w0 -eta 0.4 -iters " + iters
                    + " -early_stopping -validation_threshold 1 -disable_cv -seed 43");
        udtf.initialize(argOIs);
        udtf.initModel(udtf._params);
        for (int i = 0, n = featureVectors.size(); i < n; i++) {
            udtf.process(new Object[] {featureVectors.get(i), ys.get(i)});
        }
        udtf.finalizeTraining();

        println("Performed " + udtf._validationState.getCurrentIteration() + " iterations out of "
                + iters);
        Assert.assertEquals("Training finished earlier than expected", iters,
            udtf._validationState.getCurrentIteration());

        println("Cumulative loss: " + udtf._validationState.getCumulativeLoss());
        Assert.assertTrue("Cumulative loss should be better than " + cumulativeLoss,
            cumulativeLoss > udtf._validationState.getCumulativeLoss());
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedAdaptiveRegularizationOption() throws Exception {
        TestUtils.testGenericUDTFSerialization(FieldAwareFactorizationMachineUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-seed 43 -adaptive_regularization")},
            new Object[][] {{Arrays.asList("0:1:-2", "1:2:-1"), 1.0}});
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(FieldAwareFactorizationMachineUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-seed 43")},
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
