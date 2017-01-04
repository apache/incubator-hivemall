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

import static org.junit.Assert.assertEquals;
import hivemall.model.PredictionResult;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class KernelExpansionPassiveAggressiveUDTFTest {

    @Test
    public void testKPAEta() throws UDFArgumentException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ListObjectInspector intListOI = ObjectInspectorFactory.getStandardListObjectInspector(intOI);
        udtf.initialize(new ObjectInspector[] {intListOI, intOI});
        float loss = 0.1f;

        PredictionResult margin1 = new PredictionResult(0.5f).squaredNorm(0.05f);
        float expectedLearningRate1 = 2.0f;
        assertEquals(expectedLearningRate1, udtf.eta(loss, margin1), 1e-5f);

        PredictionResult margin2 = new PredictionResult(0.5f).squaredNorm(0.01f);
        float expectedLearningRate2 = 10.0f;
        assertEquals(expectedLearningRate2, udtf.eta(loss, margin2), 1e-5f);
    }

    @Test
    public void testKPA1Eta() throws UDFArgumentException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ListObjectInspector intListOI = ObjectInspectorFactory.getStandardListObjectInspector(intOI);
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-pa1 -c 3.0");
        udtf.initialize(new ObjectInspector[] {intListOI, intOI, param});
        float loss = 0.1f;

        PredictionResult margin1 = new PredictionResult(0.5f).squaredNorm(0.05f);
        float expectedLearningRate1 = 2.0f;
        assertEquals(expectedLearningRate1, udtf.eta(loss, margin1), 1e-5f);

        PredictionResult margin2 = new PredictionResult(0.5f).squaredNorm(0.01f);
        float expectedLearningRate2 = 3.0f;
        assertEquals(expectedLearningRate2, udtf.eta1(loss, margin2), 1e-5f);
    }

    @Test
    public void testKPA1EtaDefaultParameter() throws UDFArgumentException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ListObjectInspector intListOI = ObjectInspectorFactory.getStandardListObjectInspector(intOI);
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-pa1");
        udtf.initialize(new ObjectInspector[] {intListOI, intOI, param});
        float loss = 0.1f;

        PredictionResult margin = new PredictionResult(0.5f).squaredNorm(0.05f);
        float expectedLearningRate = 1.0f;
        assertEquals(expectedLearningRate, udtf.eta1(loss, margin), 1e-5f);
    }

    @Test
    public void testKPA2EtaWithoutParameter() throws UDFArgumentException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ListObjectInspector intListOI = ObjectInspectorFactory.getStandardListObjectInspector(intOI);
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-pa2");
        udtf.initialize(new ObjectInspector[] {intListOI, intOI, param});
        float loss = 0.1f;

        PredictionResult margin1 = new PredictionResult(0.5f).squaredNorm(0.05f);
        float expectedLearningRate1 = 0.1818181f;
        assertEquals(expectedLearningRate1, udtf.eta2(loss, margin1), 1e-5f);

        PredictionResult margin2 = new PredictionResult(0.5f).squaredNorm(0.01f);
        float expectedLearningRate2 = 0.1960784f;
        assertEquals(expectedLearningRate2, udtf.eta2(loss, margin2), 1e-5f);
    }

    @Test
    public void testKPA2EtaWithParameter() throws UDFArgumentException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ListObjectInspector intListOI = ObjectInspectorFactory.getStandardListObjectInspector(intOI);
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-pa2 -c 3.0");
        udtf.initialize(new ObjectInspector[] {intListOI, intOI, param});
        float loss = 0.1f;

        PredictionResult margin1 = new PredictionResult(0.5f).squaredNorm(0.05f);
        float expectedLearningRate1 = 0.4615384f;
        assertEquals(expectedLearningRate1, udtf.eta2(loss, margin1), 1e-5f);

        PredictionResult margin2 = new PredictionResult(0.5f).squaredNorm(0.01f);
        float expectedLearningRate2 = 0.5660377f;
        assertEquals(expectedLearningRate2, udtf.eta2(loss, margin2), 1e-5f);
    }

    @Test
    public void testNews20() throws UDFArgumentException, IOException, ParseException {
        KernelExpansionPassiveAggressiveUDTF udtf = new KernelExpansionPassiveAggressiveUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
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
            udtf.train(words, label);

            words.clear();
            line = news20.readLine();
        }

        Assert.assertTrue(Math.abs(udtf.getLoss()) < 0.1f);

        news20.close();
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = KernelExpansionPassiveAggressiveUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

}
