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

import matrix4j.matrix.Matrix;
import matrix4j.matrix.builders.CSRMatrixBuilder;
import matrix4j.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.smile.tools.TreeExportUDF.Evaluator;
import hivemall.smile.tools.TreeExportUDF.OutputType;
import hivemall.utils.codec.Base91;
import hivemall.utils.random.RandomNumberGeneratorFactory;
import smile.math.Math;
import smile.validation.LOOCV;

import java.io.IOException;
import java.text.ParseException;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;
import org.roaringbitmap.RoaringBitmap;

public class RegressionTreeTest {
    private static final boolean DEBUG = false;

    @Test
    public void testPredictDense() {

        double[][] longley = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019},
                {419.180, 282.2, 285.7, 118.734, 1956, 67.857},
                {442.769, 293.6, 279.8, 120.445, 1957, 68.169},
                {444.546, 468.1, 263.7, 121.950, 1958, 66.513},
                {482.704, 381.3, 255.2, 123.366, 1959, 68.655},
                {502.601, 393.1, 251.4, 125.368, 1960, 69.564},
                {518.173, 480.6, 257.2, 127.852, 1961, 69.331},
                {554.894, 400.7, 282.7, 130.081, 1962, 70.551}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8,
                112.6, 114.2, 115.7, 116.9};

        RoaringBitmap attrs = new RoaringBitmap();

        int n = longley.length;
        LOOCV loocv = new LOOCV(n);
        double rss = 0.0;
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(longley, loocv.train[i]);
            double[] trainy = Math.slice(y, loocv.train[i]);
            int maxLeafs = 10;
            RegressionTree tree = new RegressionTree(attrs, matrix(trainx, true), trainy, maxLeafs,
                RandomNumberGeneratorFactory.createPRNG(i));

            double r = y[loocv.test[i]] - tree.predict(longley[loocv.test[i]]);
            rss += r * r;
        }

        Assert.assertTrue("MSE = " + (rss / n), (rss / n) < 42);
    }

    @Test
    public void testPredictSparse() {

        double[][] longley = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019},
                {419.180, 282.2, 285.7, 118.734, 1956, 67.857},
                {442.769, 293.6, 279.8, 120.445, 1957, 68.169},
                {444.546, 468.1, 263.7, 121.950, 1958, 66.513},
                {482.704, 381.3, 255.2, 123.366, 1959, 68.655},
                {502.601, 393.1, 251.4, 125.368, 1960, 69.564},
                {518.173, 480.6, 257.2, 127.852, 1961, 69.331},
                {554.894, 400.7, 282.7, 130.081, 1962, 70.551}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8,
                112.6, 114.2, 115.7, 116.9};

        RoaringBitmap attrs = new RoaringBitmap();

        int n = longley.length;
        LOOCV loocv = new LOOCV(n);
        double rss = 0.0;
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(longley, loocv.train[i]);
            double[] trainy = Math.slice(y, loocv.train[i]);
            int maxLeafs = 10;
            RegressionTree tree = new RegressionTree(attrs, matrix(trainx, false), trainy, maxLeafs,
                RandomNumberGeneratorFactory.createPRNG(i));

            double r = y[loocv.test[i]] - tree.predict(longley[loocv.test[i]]);
            rss += r * r;
        }

        Assert.assertTrue("MSE = " + (rss / n), (rss / n) < 42);
    }

    @Test
    public void testSerPredict() throws HiveException {

        double[][] longley = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019},
                {419.180, 282.2, 285.7, 118.734, 1956, 67.857},
                {442.769, 293.6, 279.8, 120.445, 1957, 68.169},
                {444.546, 468.1, 263.7, 121.950, 1958, 66.513},
                {482.704, 381.3, 255.2, 123.366, 1959, 68.655},
                {502.601, 393.1, 251.4, 125.368, 1960, 69.564},
                {518.173, 480.6, 257.2, 127.852, 1961, 69.331},
                {554.894, 400.7, 282.7, 130.081, 1962, 70.551}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8,
                112.6, 114.2, 115.7, 116.9};

        RoaringBitmap attrs = new RoaringBitmap();

        int n = longley.length;
        LOOCV loocv = new LOOCV(n);
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(longley, loocv.train[i]);
            double[] trainy = Math.slice(y, loocv.train[i]);
            int maxLeafs = Integer.MAX_VALUE;
            RegressionTree tree = new RegressionTree(attrs, matrix(trainx, true), trainy, maxLeafs);

            byte[] b = tree.serialize(true);
            RegressionTree.Node node = RegressionTree.deserialize(b, b.length, true);

            double expected = tree.predict(longley[loocv.test[i]]);
            double actual = node.predict(longley[loocv.test[i]]);

            Assert.assertEquals(expected, actual, 0.d);
        }
    }

    @Test
    public void testGraphvizOutput() throws HiveException, IOException, ParseException {
        int maxLeafs = 10;
        String outputName = "predicted";

        double[][] x = {{234.289, 235.6, 159.0, 107.608, 1947, 60.323},
                {259.426, 232.5, 145.6, 108.632, 1948, 61.122},
                {258.054, 368.2, 161.6, 109.773, 1949, 60.171},
                {284.599, 335.1, 165.0, 110.929, 1950, 61.187},
                {328.975, 209.9, 309.9, 112.075, 1951, 63.221},
                {346.999, 193.2, 359.4, 113.270, 1952, 63.639},
                {365.385, 187.0, 354.7, 115.094, 1953, 64.989},
                {363.112, 357.8, 335.0, 116.219, 1954, 63.761},
                {397.469, 290.4, 304.8, 117.388, 1955, 66.019},
                {419.180, 282.2, 285.7, 118.734, 1956, 67.857},
                {442.769, 293.6, 279.8, 120.445, 1957, 68.169},
                {444.546, 468.1, 263.7, 121.950, 1958, 66.513},
                {482.704, 381.3, 255.2, 123.366, 1959, 68.655},
                {502.601, 393.1, 251.4, 125.368, 1960, 69.564},
                {518.173, 480.6, 257.2, 127.852, 1961, 69.331},
                {554.894, 400.7, 282.7, 130.081, 1962, 70.551}};

        double[] y = {83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8,
                112.6, 114.2, 115.7, 116.9};

        debugPrint(graphvizOutput(x, y, maxLeafs, true, null, outputName));
    }

    private static String graphvizOutput(double[][] x, double[] y, int maxLeafs, boolean dense,
            String[] featureNames, String outputName)
            throws IOException, HiveException, ParseException {
        RoaringBitmap attrs = new RoaringBitmap();
        RegressionTree tree = new RegressionTree(attrs, matrix(x, dense), y, maxLeafs);

        Text model = new Text(Base91.encode(tree.serialize(true)));

        Evaluator eval = new Evaluator(OutputType.graphviz, outputName, true);
        Text exported = eval.export(model, featureNames, null);

        return exported.toString();
    }

    @Nonnull
    private static Matrix matrix(@Nonnull final double[][] x, boolean dense) {
        if (dense) {
            return new RowMajorDenseMatrix2d(x, x[0].length);
        } else {
            int numRows = x.length;
            CSRMatrixBuilder builder = new CSRMatrixBuilder(1024);
            for (int i = 0; i < numRows; i++) {
                builder.nextRow(x[i]);
            }
            return builder.buildMatrix();
        }
    }

    private static void debugPrint(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }

}
