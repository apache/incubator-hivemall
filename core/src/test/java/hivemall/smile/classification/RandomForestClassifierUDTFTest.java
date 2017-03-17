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

import hivemall.classifier.KernelExpansionPassiveAggressiveUDTF;
import hivemall.utils.codec.Base91;
import hivemall.utils.lang.mutable.MutableInt;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;

public class RandomForestClassifierUDTFTest {

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

        RandomForestClassifierUDTF udtf = new RandomForestClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 49");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
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

        Assert.assertEquals(49, count.getValue());
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

        RandomForestClassifierUDTF udtf = new RandomForestClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 49");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
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

        Assert.assertEquals(49, count.getValue());
    }

    @Test
    public void testIrisSparseDenseEquals() throws IOException, ParseException, HiveException {
        String urlString = "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff";
        DecisionTree.Node denseNode = getDecisionTreeFromDenseInput(urlString);
        DecisionTree.Node sparseNode = getDecisionTreeFromSparseInput(urlString);

        URL url = new URL(urlString);
        InputStream is = new BufferedInputStream(url.openStream());
        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);

        int diff = 0;
        for (int i = 0; i < size; i++) {
            if (denseNode.predict(x[i]) != sparseNode.predict(x[i])) {
                diff++;
            }
        }

        Assert.assertTrue("large diff " + diff + " between two predictions", diff < 10);
    }

    private static DecisionTree.Node getDecisionTreeFromDenseInput(String urlString)
            throws IOException, ParseException, HiveException {
        URL url = new URL(urlString);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);
        int[] y = iris.toArray(new int[size]);

        RandomForestClassifierUDTF udtf = new RandomForestClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 1 -seed 71");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});

        final List<Double> xi = new ArrayList<Double>(x[0].length);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < x[i].length; j++) {
                xi.add(j, x[i][j]);
            }
            udtf.process(new Object[] {xi, y[i]});
            xi.clear();
        }

        final Text[] placeholder = new Text[1];
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] forward = (Object[]) input;
                placeholder[0] = (Text) forward[2];
            }
        };

        udtf.setCollector(collector);
        udtf.close();

        Text modelTxt = placeholder[0];
        Assert.assertNotNull(modelTxt);

        byte[] b = Base91.decode(modelTxt.getBytes(), 0, modelTxt.getLength());
        DecisionTree.Node node = DecisionTree.deserializeNode(b, b.length, true);
        return node;
    }

    private static DecisionTree.Node getDecisionTreeFromSparseInput(String urlString)
            throws IOException, ParseException, HiveException {
        URL url = new URL(urlString);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);

        AttributeDataset iris = arffParser.parse(is);
        int size = iris.size();
        double[][] x = iris.toArray(new double[size][]);
        int[] y = iris.toArray(new int[size]);

        RandomForestClassifierUDTF udtf = new RandomForestClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 1 -seed 71");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});

        final List<String> xi = new ArrayList<String>(x[0].length);
        for (int i = 0; i < size; i++) {
            final double[] row = x[i];
            for (int j = 0; j < row.length; j++) {
                xi.add(j + ":" + row[j]);
            }
            udtf.process(new Object[] {xi, y[i]});
            xi.clear();
        }

        final Text[] placeholder = new Text[1];
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] forward = (Object[]) input;
                placeholder[0] = (Text) forward[2];
            }
        };

        udtf.setCollector(collector);
        udtf.close();

        Text modelTxt = placeholder[0];
        Assert.assertNotNull(modelTxt);

        byte[] b = Base91.decode(modelTxt.getBytes(), 0, modelTxt.getLength());
        DecisionTree.Node node = DecisionTree.deserializeNode(b, b.length, true);
        return node;
    }

    @Test
    public void testNews20MultiClassSparse() throws IOException, ParseException, HiveException {
        final int numTrees = 1;
        RandomForestClassifierUDTF udtf = new RandomForestClassifierUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees " + numTrees);
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});


        BufferedReader news20 = readFile("news20-multiclass.gz");
        ArrayList<String> features = new ArrayList<String>();
        String line = news20.readLine();
        while (line != null) {
            StringTokenizer tokens = new StringTokenizer(line, " ");
            int label = Integer.parseInt(tokens.nextToken());
            while (tokens.hasMoreTokens()) {
                features.add(tokens.nextToken());
            }
            Assert.assertFalse(features.isEmpty());
            udtf.process(new Object[] {features, label});

            features.clear();
            line = news20.readLine();
        }
        news20.close();

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };
        udtf.setCollector(collector);
        udtf.close();

        Assert.assertEquals(numTrees, count.getValue());
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
