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
package hivemall.smile.tools;

import hivemall.TestUtils;
import matrix4j.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.regression.RegressionTree;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.lang.ArrayUtils;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.math.Math;
import smile.validation.CrossValidation;
import smile.validation.LOOCV;
import smile.validation.RMSE;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.ParseException;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;
import org.roaringbitmap.RoaringBitmap;

public class TreePredictUDFTest {
    private static final boolean DEBUG = false;

    /**
     * Test of learn method, of class DecisionTree.
     */
    @Test
    public void testIris() throws IOException, ParseException, HiveException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(4);
        AttributeDataset iris = arffParser.parse(is);
        double[][] x = iris.toArray(new double[iris.size()][]);
        int[] y = iris.toArray(new int[iris.size()]);

        int n = x.length;
        LOOCV loocv = new LOOCV(n);
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(x, loocv.train[i]);
            int[] trainy = Math.slice(y, loocv.train[i]);

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(iris.attributes());
            DecisionTree tree = new DecisionTree(attrs,
                new RowMajorDenseMatrix2d(trainx, x[0].length), trainy, 4);
            Assert.assertEquals(tree.predict(x[loocv.test[i]]),
                evalPredict(tree, x[loocv.test[i]]));
        }
    }

    @Test
    public void testCpu() throws IOException, ParseException, HiveException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/ef17aabecf0c0c5bcb69/raw/aac0575b4d43072c6f3c82d9072fdefb61892694/cpu.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(6);
        AttributeDataset data = arffParser.parse(is);
        double[] datay = data.toArray(new double[data.size()]);
        double[][] datax = data.toArray(new double[data.size()][]);

        int n = datax.length;
        int k = 10;

        CrossValidation cv = new CrossValidation(n, k);
        for (int i = 0; i < k; i++) {
            double[][] trainx = Math.slice(datax, cv.train[i]);
            double[] trainy = Math.slice(datay, cv.train[i]);
            double[][] testx = Math.slice(datax, cv.test[i]);

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(data.attributes());
            RegressionTree tree = new RegressionTree(attrs,
                new RowMajorDenseMatrix2d(trainx, trainx[0].length), trainy, 20);

            for (int j = 0; j < testx.length; j++) {
                Assert.assertEquals(tree.predict(testx[j]), evalPredict(tree, testx[j]), 1.0);
            }
        }
    }

    @Test
    public void testCpu2() throws IOException, ParseException, HiveException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/ef17aabecf0c0c5bcb69/raw/aac0575b4d43072c6f3c82d9072fdefb61892694/cpu.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(6);
        AttributeDataset data = arffParser.parse(is);
        double[] datay = data.toArray(new double[data.size()]);
        double[][] datax = data.toArray(new double[data.size()][]);

        int n = datax.length;
        int m = 3 * n / 4;
        int[] index = Math.permutate(n);

        double[][] trainx = new double[m][];
        double[] trainy = new double[m];
        for (int i = 0; i < m; i++) {
            trainx[i] = datax[index[i]];
            trainy[i] = datay[index[i]];
        }

        double[][] testx = new double[n - m][];
        double[] testy = new double[n - m];
        for (int i = m; i < n; i++) {
            testx[i - m] = datax[index[i]];
            testy[i - m] = datay[index[i]];
        }

        RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(data.attributes());
        RegressionTree tree = new RegressionTree(attrs,
            new RowMajorDenseMatrix2d(trainx, trainx[0].length), trainy, 20);
        debugPrint(String.format("RMSE = %.4f\n", rmse(tree, testx, testy)));

        for (int i = m; i < n; i++) {
            Assert.assertEquals(tree.predict(testx[i - m]), evalPredict(tree, testx[i - m]), 1.0);
        }
    }

    private static <T> double rmse(RegressionTree regression, double[][] x, double[] y) {
        final int n = x.length;
        final double[] predictions = new double[n];
        for (int i = 0; i < n; i++) {
            predictions[i] = regression.predict(x[i]);
        }
        return new RMSE().measure(y, predictions);
    }

    private static int evalPredict(DecisionTree tree, double[] x)
            throws HiveException, IOException {
        byte[] b = tree.serialize(true);
        byte[] encoded = Base91.encode(b);
        Text model = new Text(encoded);

        TreePredictUDF udf = new TreePredictUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true)});
        DeferredObject[] arguments = new DeferredObject[] {new DeferredJavaObject("model_id#1"),
                new DeferredJavaObject(model), new DeferredJavaObject(ArrayUtils.toList(x)),
                new DeferredJavaObject(true)};

        Object[] result = (Object[]) udf.evaluate(arguments);
        udf.close();
        return ((IntWritable) result[0]).get();
    }

    private static double evalPredict(RegressionTree tree, double[] x)
            throws HiveException, IOException {
        byte[] b = tree.serialize(true);
        byte[] encoded = Base91.encode(b);
        Text model = new Text(encoded);

        TreePredictUDF udf = new TreePredictUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, false)});
        DeferredObject[] arguments = new DeferredObject[] {new DeferredJavaObject("model_id#1"),
                new DeferredJavaObject(model), new DeferredJavaObject(ArrayUtils.toList(x)),
                new DeferredJavaObject(false)};

        DoubleWritable result = (DoubleWritable) udf.evaluate(arguments);
        udf.close();
        return result.get();
    }

    @Test
    public void testSerialization() throws HiveException, IOException, ParseException {
        URL url = new URL(
            "https://gist.githubusercontent.com/myui/ef17aabecf0c0c5bcb69/raw/aac0575b4d43072c6f3c82d9072fdefb61892694/cpu.arff");
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(6);
        AttributeDataset data = arffParser.parse(is);
        double[] datay = data.toArray(new double[data.size()]);
        double[][] datax = data.toArray(new double[data.size()][]);

        int n = datax.length;
        int m = 3 * n / 4;
        int[] index = Math.permutate(n);

        double[][] trainx = new double[m][];
        double[] trainy = new double[m];
        for (int i = 0; i < m; i++) {
            trainx[i] = datax[index[i]];
            trainy[i] = datay[index[i]];
        }

        double[][] testx = new double[n - m][];
        double[] testy = new double[n - m];
        for (int i = m; i < n; i++) {
            testx[i - m] = datax[index[i]];
            testy[i - m] = datay[index[i]];
        }

        RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(data.attributes());
        RegressionTree tree = new RegressionTree(attrs,
            new RowMajorDenseMatrix2d(trainx, trainx[0].length), trainy, 20);

        byte[] b = tree.serialize(true);
        byte[] encoded = Base91.encode(b);
        Text model = new Text(encoded);

        TestUtils.testGenericUDFSerialization(TreePredictUDF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, false)},
            new Object[] {"model_id#1", model, ArrayUtils.toList(testx[0])});
    }

    private static void debugPrint(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

}
