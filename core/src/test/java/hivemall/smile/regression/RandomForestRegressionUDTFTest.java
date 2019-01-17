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
package hivemall.smile.regression;

import hivemall.TestUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.hashing.MurmurHash3;
import hivemall.utils.lang.mutable.MutableInt;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

public class RandomForestRegressionUDTFTest {

    @Test
    public void testDense() throws IOException, ParseException, HiveException {
        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2};

        RandomForestRegressionUDTF udtf = new RandomForestRegressionUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 49");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector, param});

        final List<Double> xi = new ArrayList<Double>(x[0].length);
        for (int i = 0; i < x.length; i++) {
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
    public void testSparse() throws IOException, ParseException, HiveException {
        String[] featureNames = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"};

        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2};

        RandomForestRegressionUDTF udtf = new RandomForestRegressionUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 49");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector, param});

        final List<String> xi = new ArrayList<String>(x[0].length);
        for (int i = 0; i < x.length; i++) {
            double[] row = x[i];
            for (int j = 0; j < row.length; j++) {
                xi.add(mhash(featureNames[j]) + ":" + row[j]);
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
    public void testSparseDenseEquals() throws IOException, ParseException, HiveException {
        RegressionTree.Node denseNode = getRegressionTreeFromDenseInput();
        RegressionTree.Node sparseNode = getRegressionTreeFromSparseInput();

        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        int diff = 0;
        for (int i = 0; i < x.length; i++) {
            if (denseNode.predict(x[i]) != sparseNode.predict(x[i])) {
                diff++;
            }
        }

        Assert.assertTrue("large diff " + diff + " between two predictions", diff < 10);
    }

    private static RegressionTree.Node getRegressionTreeFromDenseInput()
            throws IOException, ParseException, HiveException {
        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2};

        RandomForestRegressionUDTF udtf = new RandomForestRegressionUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 1 -seed 71");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector, param});

        final List<Double> xi = new ArrayList<Double>(x[0].length);
        for (int i = 0; i < x.length; i++) {
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
        RegressionTree.Node node = RegressionTree.deserialize(b, b.length, true);
        return node;
    }

    private static RegressionTree.Node getRegressionTreeFromSparseInput()
            throws IOException, ParseException, HiveException {
        String[] featureNames = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"};

        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2};

        RandomForestRegressionUDTF udtf = new RandomForestRegressionUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 1 -seed 71");
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector, param});

        final List<String> xi = new ArrayList<String>(x[0].length);
        for (int i = 0; i < x.length; i++) {
            final double[] row = x[i];
            for (int j = 0; j < row.length; j++) {
                xi.add(mhash(featureNames[j]) + ":" + row[j]);
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
        RegressionTree.Node node = RegressionTree.deserialize(b, b.length, true);
        return node;
    }

    @Test
    public void testSerialization() throws HiveException, IOException, ParseException {
        String[] featureNames = {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"};

        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2};

        final Object[][] rows = new Object[x.length][2];
        for (int i = 0; i < x.length; i++) {
            double[] row = x[i];
            final List<String> xi = new ArrayList<String>(x[0].length);
            for (int j = 0; j < row.length; j++) {
                xi.add(mhash(featureNames[j]) + ":" + row[j]);
            }
            rows[i][0] = xi;
            rows[i][1] = y[i];
        }

        TestUtils.testGenericUDTFSerialization(RandomForestRegressionUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-trees 49")},
            rows);
    }

    private static int mhash(@Nonnull final String word) {
        final int n = 16777217; // 2^24
        int r = MurmurHash3.murmurhash3_x86_32(word, 0, word.length(), 0x9747b28c) % n;
        if (r < 0) {
            r += n;
        }
        return r + 1;
    }

}
