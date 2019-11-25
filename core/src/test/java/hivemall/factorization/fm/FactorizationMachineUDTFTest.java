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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import hivemall.TestUtils;
import hivemall.factorization.fm.FMStringFeatureMapModel;
import hivemall.factorization.fm.FactorizationMachineModel;
import hivemall.factorization.fm.FactorizationMachineUDTF;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class FactorizationMachineUDTFTest {
    private static final boolean DEBUG = false;
    private static final int ITERATIONS = 50;

    @Test
    public void testSGD() throws HiveException, IOException {
        println("SGD test");
        FactorizationMachineUDTF udtf = new FactorizationMachineUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-factors 5 -min 1 -max 5 -iters 1 -init_v gaussian -eta0 0.01 -seed 31")};

        udtf.initialize(argOIs);
        FactorizationMachineModel model = udtf.initModel(udtf._params);

        Assert.assertFalse(udtf._params.l2norm);
        Assert.assertTrue("Actual class: " + model.getClass().getName(),
            model instanceof FMStringFeatureMapModel);

        double loss = 0.d;
        double cumul = 0.d;
        for (int trainingIteration = 1; trainingIteration <= ITERATIONS; ++trainingIteration) {
            BufferedReader data = readFile("5107786.txt.gz");
            loss = udtf._cvState.getCumulativeLoss();
            int trExamples = 0;
            String line = data.readLine();
            while (line != null) {
                StringTokenizer tokenizer = new StringTokenizer(line, " ");
                double y = Double.parseDouble(tokenizer.nextToken());
                List<String> features = new ArrayList<String>();
                while (tokenizer.hasMoreTokens()) {
                    String f = tokenizer.nextToken();
                    features.add(f);
                }
                udtf.process(new Object[] {features, y});
                trExamples++;
                line = data.readLine();
            }
            cumul = udtf._cvState.getCumulativeLoss();
            loss = (cumul - loss) / trExamples;
            println(
                trainingIteration + " " + loss + " " + cumul / (trainingIteration * trExamples));
            data.close();
        }

        Assert.assertTrue("Loss was greater than 0.1: " + loss, loss <= 0.1);
    }

    @Test
    public void testAdaptiveRegularization() throws HiveException, IOException {
        println("Adaptive regularization test");

        final String options = "-factors 5 -min 1 -max 5 -init_v gaussian -eta0 0.01 -seed 31 ";

        FactorizationMachineUDTF udtf = new FactorizationMachineUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, options)};

        udtf.initialize(argOIs);

        BufferedReader data = readFile("5107786.txt.gz");
        List<List<String>> featureVectors = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        String line = data.readLine();
        while (line != null) {
            StringTokenizer tokenizer = new StringTokenizer(line, " ");
            double y = Double.parseDouble(tokenizer.nextToken());
            List<String> features = new ArrayList<String>();
            while (tokenizer.hasMoreTokens()) {
                String f = tokenizer.nextToken();
                features.add(f);
            }
            udtf.process(new Object[] {features, y});
            featureVectors.add(features);
            ys.add(y);
            line = data.readLine();
        }
        udtf.finalizeTraining();
        data.close();

        double loss = udtf._cvState.getAverageLoss(featureVectors.size());
        println("Average loss without adaptive regularization: " + loss);

        // train with adaptive regularization
        udtf = new FactorizationMachineUDTF();
        argOIs[2] = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            options + "-adaptive_regularization -validation_threshold 1");
        udtf.initialize(argOIs);
        udtf.initModel(udtf._params);
        for (int i = 0, n = featureVectors.size(); i < n; i++) {
            udtf.process(new Object[] {featureVectors.get(i), ys.get(i)});
        }
        udtf.finalizeTraining();

        double loss_adareg = udtf._cvState.getAverageLoss(featureVectors.size());
        println("Average loss with adaptive regularization: " + loss_adareg);
        Assert.assertTrue("Adaptive regularization should achieve lower loss", loss > loss_adareg);
    }

    @Test
    public void testEarlyStopping() throws HiveException, IOException {
        println("Early stopping test");

        int iters = 20;

        // train with 20 iterations
        FactorizationMachineUDTF udtf = new FactorizationMachineUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-factors 5 -min 1 -max 5 -init_v gaussian -eta0 0.002 -seed 31 -iters " + iters
                            + " -early_stopping -validation_threshold 1 -disable_cv")};

        udtf.initialize(argOIs);

        BufferedReader data = readFile("5107786.txt.gz");
        List<List<String>> featureVectors = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        String line = data.readLine();
        while (line != null) {
            StringTokenizer tokenizer = new StringTokenizer(line, " ");
            double y = Double.parseDouble(tokenizer.nextToken());
            List<String> features = new ArrayList<String>();
            while (tokenizer.hasMoreTokens()) {
                String f = tokenizer.nextToken();
                features.add(f);
            }
            udtf.process(new Object[] {features, y});
            featureVectors.add(features);
            ys.add(y);
            line = data.readLine();
        }
        udtf.finalizeTraining();
        data.close();

        double loss = udtf._validationState.getAverageLoss(featureVectors.size());
        Assert.assertTrue(
            "Training seems to be failed because average loss is greater than 0.1: " + loss,
            loss <= 0.1);

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
        udtf = new FactorizationMachineUDTF();
        argOIs[2] = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            "-factors 5 -min 1 -max 5 -init_v gaussian -eta0 0.002 -seed 31 -iters " + iters
                    + " -early_stopping -validation_threshold 1 -disable_cv");
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

    @Test
    public void testEnableL2Norm() throws HiveException, IOException {
        FactorizationMachineUDTF udtf = new FactorizationMachineUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-factors 5 -min 1 -max 5 -iters 1 -init_v gaussian -eta0 0.01 -seed 31 -l2norm")};

        udtf.initialize(argOIs);
        udtf.initModel(udtf._params);
        Assert.assertTrue(udtf._params.l2norm);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(FactorizationMachineUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-factors 5 -min 1 -max 5 -iters 1 -init_v gaussian -eta0 0.01 -seed 31")},
            new Object[][] {{Arrays.asList("1:-2", "2:-1"), 1.0}});
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = FactorizationMachineUDTFTest.class.getResourceAsStream(fileName);
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
