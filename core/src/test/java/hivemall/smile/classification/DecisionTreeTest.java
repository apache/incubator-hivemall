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

import static org.junit.Assert.assertEquals;
import hivemall.matrix.Matrix;
import hivemall.matrix.builders.CSRMatrixBuilder;
import hivemall.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.smile.classification.DecisionTree.Node;
import hivemall.smile.data.Attribute;
import hivemall.smile.utils.SmileExtUtils;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.ParseException;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Test;

import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.math.Math;
import smile.validation.LOOCV;

public class DecisionTreeTest {
    private static final boolean DEBUG = false;

    /**
     * Test of learn method, of class DecisionTree.
     * 
     * @throws ParseException
     * @throws IOException
     */
    @Test
    public void testWeather() throws IOException, ParseException {
        int responseIndex = 4;
        int numLeafs = 3;

        // dense matrix
        int error = run(
            "https://gist.githubusercontent.com/myui/2c9df50db3de93a71b92/raw/3f6b4ecfd4045008059e1a2d1c4064fb8a3d372a/weather.nominal.arff",
            responseIndex, numLeafs, true);
        assertEquals(5, error);

        // sparse matrix
        error = run(
            "https://gist.githubusercontent.com/myui/2c9df50db3de93a71b92/raw/3f6b4ecfd4045008059e1a2d1c4064fb8a3d372a/weather.nominal.arff",
            responseIndex, numLeafs, false);
        assertEquals(5, error);
    }

    @Test
    public void testIris() throws IOException, ParseException {
        int responseIndex = 4;
        int numLeafs = Integer.MAX_VALUE;
        int error = run(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs, true);
        assertEquals(8, error);

        // sparse
        error = run(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs, false);
        assertEquals(8, error);
    }

    @Test
    public void testIrisDepth4() throws IOException, ParseException {
        int responseIndex = 4;
        int numLeafs = 4;
        int error = run(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs, true);
        assertEquals(7, error);

        // sparse 
        error = run(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs, false);
        assertEquals(7, error);
    }

    private static int run(String datasetUrl, int responseIndex, int numLeafs, boolean dense)
            throws IOException, ParseException {
        URL url = new URL(datasetUrl);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(responseIndex);

        AttributeDataset ds = arffParser.parse(is);
        double[][] x = ds.toArray(new double[ds.size()][]);
        int[] y = ds.toArray(new int[ds.size()]);

        int n = x.length;
        LOOCV loocv = new LOOCV(n);
        int error = 0;
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(x, loocv.train[i]);
            int[] trainy = Math.slice(y, loocv.train[i]);

            Attribute[] attrs = SmileExtUtils.convertAttributeTypes(ds.attributes());
            smile.math.Random rand = new smile.math.Random(i);
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, dense), trainy, numLeafs,
                rand);
            if (y[loocv.test[i]] != tree.predict(x[loocv.test[i]])) {
                error++;
            }
        }

        debugPrint("Decision Tree error = " + error);
        return error;
    }

    @Test
    public void testIrisSerializedObj() throws IOException, ParseException, HiveException {
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

            Attribute[] attrs = SmileExtUtils.convertAttributeTypes(iris.attributes());
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, true), trainy, 4);

            byte[] b = tree.predictSerCodegen(false);
            Node node = DecisionTree.deserializeNode(b, b.length, false);
            assertEquals(tree.predict(x[loocv.test[i]]), node.predict(x[loocv.test[i]]));
        }
    }

    @Test
    public void testIrisSerializeObjCompressed() throws IOException, ParseException, HiveException {
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

            Attribute[] attrs = SmileExtUtils.convertAttributeTypes(iris.attributes());
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, true), trainy, 4);

            byte[] b1 = tree.predictSerCodegen(true);
            byte[] b2 = tree.predictSerCodegen(false);
            Assert.assertTrue("b1.length = " + b1.length + ", b2.length = " + b2.length,
                b1.length < b2.length);
            Node node = DecisionTree.deserializeNode(b1, b1.length, true);
            assertEquals(tree.predict(x[loocv.test[i]]), node.predict(x[loocv.test[i]]));
        }
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
