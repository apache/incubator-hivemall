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

import hivemall.TestUtils;
import hivemall.model.FeatureValue;
import hivemall.utils.math.MathUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class KernelExpansionPassiveAggressiveUDTFTest {

    @Test
    public void testNews20() throws IOException, ParseException, HiveException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI =
                ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        udtf.initialize(new ObjectInspector[] {stringListOI, intOI});

        BufferedReader news20 = readFile("news20-small.binary.gz");
        ArrayList<String> words = new ArrayList<String>();
        String line = news20.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            int label = Integer.parseInt(tokens.nextToken());
            while (tokens.hasMoreTokens()) {
                words.add(tokens.nextToken());
            }
            Assert.assertFalse(words.isEmpty());
            udtf.process(new Object[] {words, label});

            words.clear();
            line = news20.readLine();
        }

        Assert.assertTrue(Math.abs(udtf.getLoss()) < 0.25f);

        news20.close();
    }

    public void test_a9a() throws IOException, ParseException, HiveException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI =
                ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-c 0.01");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});

        final ArrayList<String> words = new ArrayList<String>();
        BufferedReader trainData = readFile("a9a.gz");
        String line = trainData.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            String labelStr = tokens.nextToken();
            final int label;
            if ("+1".equals(labelStr)) {
                label = 1;
            } else if ("-1".equals(labelStr)) {
                label = -1;
            } else {
                throw new IllegalStateException("Illegal label: " + labelStr);
            }
            while (tokens.hasMoreTokens()) {
                words.add(tokens.nextToken());
            }
            Assert.assertFalse(words.isEmpty());
            udtf.process(new Object[] {words, label});

            words.clear();
            line = trainData.readLine();
        }
        trainData.close();

        int numTests = 0;
        int numCorrect = 0;

        BufferedReader testData = readFile("a9a.t.gz");
        line = testData.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            String labelStr = tokens.nextToken();
            final int actual;
            if ("+1".equals(labelStr)) {
                actual = 1;
            } else if ("-1".equals(labelStr)) {
                actual = -1;
            } else {
                throw new IllegalStateException("Illegal label: " + labelStr);
            }
            while (tokens.hasMoreTokens()) {
                words.add(tokens.nextToken());
            }
            Assert.assertFalse(words.isEmpty());

            FeatureValue[] features = udtf.parseFeatures(words);
            float score = udtf.predict(features);
            int predicted = MathUtils.sign(score);

            if (predicted == actual) {
                ++numCorrect;
            }
            ++numTests;

            words.clear();
            line = testData.readLine();
        }
        testData.close();

        float accuracy = numCorrect / (float) numTests;
        Assert.assertTrue(accuracy > 0.82f);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(KernelExpansionPassiveAggressiveUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector},
            new Object[][] {{Arrays.asList("1:-2", "2:-1"), 0}});
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is =
                KernelExpansionPassiveAggressiveUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

}
