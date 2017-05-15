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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import hivemall.fm.FactorizationMachineUDTFTest;

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

public class GeneralRegressionUDTFTest {
    private static final boolean DEBUG = false;

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedOptimizer() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-opt UnsupportedOpt");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedLossFunction() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss UnsupportedLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidLossFunction() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-loss HingeLoss");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedRegularization() throws Exception {
        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector intOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reg UnsupportedReg");

        udtf.initialize(new ObjectInspector[] {stringListOI, intOI, params});
    }

    @Test
    public void testSGD() throws IOException, ParseException, HiveException {
        int nIter = 10;

        GeneralRegressionUDTF udtf = new GeneralRegressionUDTF();
        ObjectInspector floatOI = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector stringOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        ListObjectInspector stringListOI = ObjectInspectorFactory.getStandardListObjectInspector(stringOI);
        ObjectInspector params = ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                "-opt SGD -loss squaredloss -reg L2 -lambda 1e-3 -eta fixed -eta0 1e-6");

        udtf.initialize(new ObjectInspector[] {stringListOI, floatOI, params});

        BufferedReader news20 = readFile("5107786.txt.gz");
        ArrayList<String> features = new ArrayList<String>();
        ArrayList<ArrayList<String>> featuresList = new ArrayList<ArrayList<String>>();
        ArrayList<Float> ys =  new ArrayList<Float>();
        String line = news20.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            float y = Float.parseFloat(tokens.nextToken()) / 5.f;
            while (tokens.hasMoreTokens()) {
                features.add(tokens.nextToken());
            }
            Assert.assertFalse(features.isEmpty());
            udtf.process(new Object[] {features, y});

            ys.add(y);
            featuresList.add((ArrayList) features.clone());

            features.clear();
            line = news20.readLine();
        }
        news20.close();

        // perform SGD iterations
        for (int it = 1; it < nIter; it++) {
            for (int i = 0, size = featuresList.size(); i < size; i++) {
                features = featuresList.get(i);
                float y = ys.get(i);
                udtf.process(new Object[] {features, y});
            }
        }

        float accum = 0.f;

        for (int i = 0, size = featuresList.size(); i < size; i++) {
            features = featuresList.get(i);
            float y = ys.get(i);

            float predicted = udtf.predict(udtf.parseFeatures(features));
            println("Predicted: " + predicted + ", Actual: " + y);

            accum += Math.abs(y - predicted);
        }

        float err = accum / ys.size();
        println("Mean absolute error: " + err);
        Assert.assertTrue(err < 2.5f);
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        // use testing resource of FMs
        InputStream is = FactorizationMachineUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
