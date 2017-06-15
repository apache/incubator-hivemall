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
package hivemall.classifier;

import hivemall.utils.math.MathUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class GeneralClassifierUDTFTest {
    private static final boolean DEBUG = false;

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedOptimizer() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-opt UnsupportedOpt");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedLossFunction() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss UnsupportedLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedRegularization() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reg UnsupportedReg");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    private void run(@Nonnull String options) throws Exception {
        println(options);

        ArrayList<List<String>> samplesList = new ArrayList<List<String>>();
        samplesList.add(Arrays.asList("1:-2", "2:-1"));
        samplesList.add(Arrays.asList("1:-1", "2:-1"));
        samplesList.add(Arrays.asList("1:-1", "2:-2"));
        samplesList.add(Arrays.asList("1:1", "2:1"));
        samplesList.add(Arrays.asList("1:1", "2:2"));
        samplesList.add(Arrays.asList("1:2", "2:1"));

        int[] labels = new int[] {0, 0, 0, 1, 1, 1};

        int maxIter = 512;

        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, options);

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});

        double cumLossPrev = Double.MAX_VALUE;
        double cumLoss = 0.d;
        int it = 0;
        while ((it < maxIter) && (Math.abs(cumLoss - cumLossPrev) > 1e-3f)) {
            cumLossPrev = cumLoss;
            udtf.resetCumulativeLoss();
            for (int i = 0, size = samplesList.size(); i < size; i++) {
                udtf.process(new Object[] {samplesList.get(i), labels[i]});
            }
            cumLoss = udtf.getCumulativeLoss();
            println("Iter: " + ++it + ", Cumulative loss: " + cumLoss);
        }
        Assert.assertTrue(cumLoss / samplesList.size() < 0.5d);

        int numTests = 0;
        int numCorrect = 0;

        for (int i = 0, size = samplesList.size(); i < size; i++) {
            int label = labels[i];

            float score = udtf.predict(udtf.parseFeatures(samplesList.get(i)));
            int predicted = score > 0.f ? 1 : 0;

            println("Score: " + score + ", Predicted: " + predicted + ", Actual: " + label);

            if (predicted == label) {
                ++numCorrect;
            }
            ++numTests;
        }

        float accuracy = numCorrect / (float) numTests;
        println("Accuracy: " + accuracy);
        Assert.assertTrue(accuracy == 1.f);
    }

    @Test
    public void test() throws Exception {
        String[] optimizers = new String[] {"SGD", "AdaDelta", "AdaGrad", "Adam"};
        String[] regularizations = new String[] {"NO", "L1", "L2", "ElasticNet", "RDA"};
        String[] lossFunctions = new String[] {"HingeLoss", "LogLoss", "SquaredHingeLoss",
                "ModifiedHuberLoss", "SquaredLoss", "QuantileLoss", "EpsilonInsensitiveLoss",
                "SquaredEpsilonInsensitiveLoss", "HuberLoss"};

        for (String opt : optimizers) {
            for (String reg : regularizations) {
                if (reg == "RDA" && opt != "AdaGrad") {
                    continue;
                }

                for (String loss : lossFunctions) {
                    String options = "-opt " + opt + " -reg " + reg + " -loss " + loss;

                    // sparse
                    run(options);

                    if (opt != "AdaGrad") {
                        options += " -mini_batch 2";
                        run(options);
                    }

                    // dense
                    options += " -dense";
                    run(options);
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testNews20() throws IOException, ParseException, HiveException {
        int nIter = 10;

        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            "-opt SGD -loss logloss -reg L2 -lambda 0.1");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});

        BufferedReader news20 = readFile("news20-small.binary.gz");
        ArrayList<Integer> labels = new ArrayList<Integer>();
        ArrayList<String> words = new ArrayList<String>();
        ArrayList<ArrayList<String>> wordsList = new ArrayList<ArrayList<String>>();
        String line = news20.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            int label = Integer.parseInt(tokens.nextToken());
            while (tokens.hasMoreTokens()) {
                words.add(tokens.nextToken());
            }
            Assert.assertFalse(words.isEmpty());
            udtf.process(new Object[] {words, label});

            labels.add(label);
            wordsList.add((ArrayList<String>) words.clone());

            words.clear();
            line = news20.readLine();
        }
        news20.close();

        // perform SGD iterations
        for (int it = 1; it < nIter; it++) {
            for (int i = 0, size = wordsList.size(); i < size; i++) {
                words = wordsList.get(i);
                int label = labels.get(i);
                udtf.process(new Object[] {words, label});
            }
        }

        int numTests = 0;
        int numCorrect = 0;

        for (int i = 0, size = wordsList.size(); i < size; i++) {
            words = wordsList.get(i);
            int label = labels.get(i);

            float score = udtf.predict(udtf.parseFeatures(words));
            int predicted = MathUtils.sign(score);

            println("Score: " + score + ", Predicted: " + predicted + ", Actual: " + label);

            if (predicted == label) {
                ++numCorrect;
            }
            ++numTests;
        }

        float accuracy = numCorrect / (float) numTests;
        println("Accuracy: " + accuracy);
        Assert.assertTrue(accuracy > 0.8f);
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = GeneralClassifierUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }
}
