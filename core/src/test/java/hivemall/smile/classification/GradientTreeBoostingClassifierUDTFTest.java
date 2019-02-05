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
package hivemall.smile.classification;

import hivemall.TestUtils;
import hivemall.classifier.KernelExpansionPassiveAggressiveUDTF;
import hivemall.utils.lang.mutable.MutableInt;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;

import javax.annotation.Nonnull;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class GradientTreeBoostingClassifierUDTFTest {

    @Test
    public void testIrisDense() throws IOException, ParseException, HiveException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);
        int[] y = iris.toArray(new int[size]);

        GradientTreeBoostingClassifierUDTF udtf = new GradientTreeBoostingClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 490");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});

        final List<Double> xi = new ArrayList<Double>(x[0].length);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < x[i].length; j++) {
                xi.add(j, x[i][j]);
            }
            udtf.process(new Object[] {xi, y[i]});
            xi.clear();
        }

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };

        udtf.setCollector(collector);
        udtf.close();

        Assert.assertEquals(490, count.getValue());
    }

    @Test
    public void testIrisSparse() throws IOException, ParseException, HiveException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);
        int[] y = iris.toArray(new int[size]);

        GradientTreeBoostingClassifierUDTF udtf = new GradientTreeBoostingClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 490");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});

        final List<String> xi = new ArrayList<String>(x[0].length);
        for (int i = 0; i < size; i++) {
            double[] row = x[i];
            for (int j = 0; j < row.length; j++) {
                xi.add(j + ":" + row[j]);
            }
            udtf.process(new Object[] {xi, y[i]});
            xi.clear();
        }

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };

        udtf.setCollector(collector);
        udtf.close();

        Assert.assertEquals(490, count.getValue());
    }

    @Test
    public void testSerialization() throws HiveException, IOException, ParseException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);
        int[] y = iris.toArray(new int[size]);

        final Object[][] rows = new Object[size][2];
        for (int i = 0; i < size; i++) {
            double[] row = x[i];
            final List<String> xi = new ArrayList<String>(x[0].length);
            for (int j = 0; j < row.length; j++) {
                xi.add(j + ":" + row[j]);
            }
            rows[i][0] = xi;
            rows[i][1] = y[i];
        }

        TestUtils.testGenericUDTFSerialization(GradientTreeBoostingClassifierUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 490")},
            rows);
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = KernelExpansionPassiveAggressiveUDTF.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }
}
