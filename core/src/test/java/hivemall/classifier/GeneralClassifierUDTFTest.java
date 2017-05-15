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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import hivemall.utils.math.MathUtils;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Assert;
import org.junit.Test;

import javax.annotation.Nonnull;

public class GeneralClassifierUDTFTest {
    private static final boolean DEBUG = false;

    @Test(expected = IllegalArgumentException.class)
    public void testUnsupportedOptimizer() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-opt UnsupportedOpt");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = IllegalArgumentException.class)
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
    public void testInvalidLossFunction() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss SquaredLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = IllegalArgumentException.class)
    public void testUnsupportedRegularization() throws Exception {
        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reg UnsupportedReg");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test
    public void testSGDNews20() throws IOException, ParseException, HiveException {
        int nIter = 10;

        GeneralClassifierUDTF udtf = new GeneralClassifierUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-opt SGD -loss logloss -reg L2");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});

        BufferedReader news20 = readFile("news20-small.binary.gz");
        ArrayList<Integer> labels =  new ArrayList<Integer>();
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
            wordsList.add((ArrayList) words.clone());

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

            println("Predicted: " + predicted + ", Actual: " + label);

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
