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

import matrix4j.matrix.Matrix;
import matrix4j.matrix.builders.CSRMatrixBuilder;
import matrix4j.matrix.dense.RowMajorDenseMatrix2d;
import matrix4j.vector.DenseVector;
import hivemall.smile.classification.DecisionTree.Node;
import hivemall.smile.classification.DecisionTree.SplitRule;
import hivemall.smile.tools.TreeExportUDF.Evaluator;
import hivemall.smile.tools.TreeExportUDF.OutputType;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.StringUtils;
import hivemall.utils.math.MathUtils;
import hivemall.utils.random.PRNG;
import hivemall.utils.random.RandomNumberGeneratorFactory;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.parser.ArffParser;
import smile.data.parser.DelimitedTextParser;
import smile.math.Math;
import smile.validation.LOOCV;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.ParseException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Random;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;
import org.roaringbitmap.RoaringBitmap;

public class DecisionTreeTest {
    private static final boolean DEBUG = false;

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
    public void testIrisSparseDenseEquals() throws IOException, ParseException {
        int responseIndex = 4;
        int numLeafs = Integer.MAX_VALUE;
        runAndCompareSparseAndDense(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs);
    }

    @Test
    public void testIrisTracePredict() throws IOException, ParseException {
        int responseIndex = 4;
        int numLeafs = Integer.MAX_VALUE;
        runTracePredict(
            "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff",
            responseIndex, numLeafs);
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

    @Test
    public void testGraphvizOutputIris() throws IOException, ParseException, HiveException {
        String datasetUrl =
                "https://gist.githubusercontent.com/myui/143fa9d05bd6e7db0114/raw/500f178316b802f1cade6e3bf8dc814a96e84b1e/iris.arff";
        int responseIndex = 4;
        int numLeafs = 4;
        boolean dense = true;
        String outputName = "class";
        String[] featureNames =
                new String[] {"sepallength", "sepalwidth", "petallength", "petalwidth"};
        String[] classNames = new String[] {"setosa", "versicolor", "virginica"};

        debugPrint(graphvizOutput(datasetUrl, responseIndex, numLeafs, dense, featureNames,
            classNames, outputName));

        featureNames = null;
        classNames = null;
        outputName = null;
        debugPrint(graphvizOutput(datasetUrl, responseIndex, numLeafs, dense, featureNames,
            classNames, outputName));
    }

    @Test
    public void testGraphvizOutputWeather() throws IOException, ParseException, HiveException {
        String datasetUrl =
                "https://gist.githubusercontent.com/myui/2c9df50db3de93a71b92/raw/3f6b4ecfd4045008059e1a2d1c4064fb8a3d372a/weather.nominal.arff";
        int responseIndex = 4;
        int numLeafs = 3;
        boolean dense = true;
        String[] featureNames = new String[] {"outlook", "temperature", "humidity", "windy"};
        String[] classNames = new String[] {"yes", "no"};
        String outputName = "play";

        debugPrint(graphvizOutput(datasetUrl, responseIndex, numLeafs, dense, featureNames,
            classNames, outputName));

        featureNames = null;
        classNames = null;
        debugPrint(graphvizOutput(datasetUrl, responseIndex, numLeafs, dense, featureNames,
            classNames, outputName));
    }

    private static String graphvizOutput(String datasetUrl, int responseIndex, int numLeafs,
            boolean dense, String[] featureNames, String[] classNames, String outputName)
            throws IOException, HiveException, ParseException {
        URL url = new URL(datasetUrl);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(responseIndex);

        AttributeDataset ds = arffParser.parse(is);
        double[][] x = ds.toArray(new double[ds.size()][]);
        int[] y = ds.toArray(new int[ds.size()]);

        RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(ds.attributes());
        DecisionTree tree = new DecisionTree(attrs, matrix(x, dense), y, numLeafs,
            RandomNumberGeneratorFactory.createPRNG(31));

        Text model = new Text(Base91.encode(tree.serialize(true)));

        Evaluator eval = new Evaluator(OutputType.graphviz, outputName, false);
        Text exported = eval.export(model, featureNames, classNames);

        return exported.toString();
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

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(ds.attributes());
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, dense), trainy, numLeafs,
                RandomNumberGeneratorFactory.createPRNG(i));
            if (y[loocv.test[i]] != tree.predict(x[loocv.test[i]])) {
                error++;
            }
        }

        debugPrint("Decision Tree error = " + error);
        return error;
    }

    private static void runAndCompareSparseAndDense(String datasetUrl, int responseIndex,
            int numLeafs) throws IOException, ParseException {
        URL url = new URL(datasetUrl);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(responseIndex);

        AttributeDataset ds = arffParser.parse(is);
        double[][] x = ds.toArray(new double[ds.size()][]);
        int[] y = ds.toArray(new int[ds.size()]);

        int n = x.length;
        LOOCV loocv = new LOOCV(n);
        for (int i = 0; i < n; i++) {
            double[][] trainx = Math.slice(x, loocv.train[i]);
            int[] trainy = Math.slice(y, loocv.train[i]);

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(ds.attributes());
            DecisionTree dtree = new DecisionTree(attrs, matrix(trainx, true), trainy, numLeafs,
                RandomNumberGeneratorFactory.createPRNG(i));
            DecisionTree stree = new DecisionTree(attrs, matrix(trainx, false), trainy, numLeafs,
                RandomNumberGeneratorFactory.createPRNG(i));
            Assert.assertEquals(dtree.predict(x[loocv.test[i]]), stree.predict(x[loocv.test[i]]));
            Assert.assertEquals(dtree.toString(), stree.toString());
        }
    }

    private static void runTracePredict(String datasetUrl, int responseIndex, int numLeafs)
            throws IOException, ParseException {
        URL url = new URL(datasetUrl);
        InputStream is = new BufferedInputStream(url.openStream());

        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(responseIndex);

        AttributeDataset ds = arffParser.parse(is);
        final Attribute[] attrs = ds.attributes();
        final Attribute targetAttr = ds.response();

        double[][] x = ds.toArray(new double[ds.size()][]);
        int[] y = ds.toArray(new int[ds.size()]);

        Random rnd = new Random(43L);
        int numTrain = (int) (x.length * 0.7);
        int[] index = ArrayUtils.shuffle(MathUtils.permutation(x.length), rnd);
        int[] cvTrain = Arrays.copyOf(index, numTrain);
        int[] cvTest = Arrays.copyOfRange(index, numTrain, index.length);

        double[][] trainx = Math.slice(x, cvTrain);
        int[] trainy = Math.slice(y, cvTrain);
        double[][] testx = Math.slice(x, cvTest);

        DecisionTree tree = new DecisionTree(SmileExtUtils.convertAttributeTypes(attrs),
            matrix(trainx, false), trainy, numLeafs, RandomNumberGeneratorFactory.createPRNG(43L));

        final LinkedHashMap<String, Double> map = new LinkedHashMap<>();
        final StringBuilder buf = new StringBuilder();
        for (int i = 0; i < testx.length; i++) {
            final DenseVector test = new DenseVector(testx[i]);
            tree.predict(test, new PredictionHandler() {

                @Override
                public void visitBranch(Operator op, int splitFeatureIndex, double splitFeature,
                        double splitValue) {
                    buf.append(attrs[splitFeatureIndex].name);
                    buf.append(" [" + splitFeature + "] ");
                    buf.append(op);
                    buf.append(' ');
                    buf.append(splitValue);
                    buf.append('\n');

                    map.put(attrs[splitFeatureIndex].name + " [" + splitFeature + "] " + op,
                        splitValue);
                }

                @Override
                public void visitLeaf(int output, double[] posteriori) {
                    buf.append(targetAttr.toString(output));
                }
            });

            Assert.assertTrue(buf.length() > 0);
            Assert.assertFalse(map.isEmpty());

            StringUtils.clear(buf);
            map.clear();
        }

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

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(iris.attributes());
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, true), trainy, 4);

            byte[] b = tree.serialize(false);
            Node node = DecisionTree.deserialize(b, b.length, false);
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

            RoaringBitmap attrs = SmileExtUtils.convertAttributeTypes(iris.attributes());
            DecisionTree tree = new DecisionTree(attrs, matrix(trainx, true), trainy, 4);

            byte[] b1 = tree.serialize(true);
            byte[] b2 = tree.serialize(false);
            Assert.assertTrue("b1.length = " + b1.length + ", b2.length = " + b2.length,
                b1.length < b2.length);
            Node node = DecisionTree.deserialize(b1, b1.length, true);
            assertEquals(tree.predict(x[loocv.test[i]]), node.predict(x[loocv.test[i]]));
        }
    }

    @Test
    public void testTitanicPruning() throws IOException, ParseException {
        String datasetUrl =
                "https://gist.githubusercontent.com/myui/7cd82c443db84ba7e7add1523d0247a9/raw/f2d3e3051b0292577e8c01a1759edabaa95c5781/titanic_train.tsv";

        URL url = new URL(datasetUrl);
        InputStream is = new BufferedInputStream(url.openStream());

        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setColumnNames(true);
        parser.setDelimiter(",");
        parser.setResponseIndex(new NominalAttribute("survived"), 0);

        AttributeDataset train = parser.parse("titanic train", is);
        double[][] x_ = train.toArray(new double[train.size()][]);
        int[] y = train.toArray(new int[train.size()]);

        // pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked
        // C,C,C,Q,Q,Q,C,Q,C,C
        RoaringBitmap nominalAttrs = new RoaringBitmap();
        nominalAttrs.add(0);
        nominalAttrs.add(1);
        nominalAttrs.add(2);
        nominalAttrs.add(6);
        nominalAttrs.add(8);
        nominalAttrs.add(9);

        int columns = x_[0].length;
        Matrix x = new RowMajorDenseMatrix2d(x_, columns);
        int numVars = (int) Math.ceil(Math.sqrt(columns));
        int maxDepth = Integer.MAX_VALUE;
        int maxLeafs = Integer.MAX_VALUE;
        int minSplits = 2;
        int minLeafSize = 1;
        int[] samples = null;
        PRNG rand = RandomNumberGeneratorFactory.createPRNG(43L);

        final String[] featureNames = new String[] {"pclass", "name", "sex", "age", "sibsp",
                "parch", "ticket", "fare", "cabin", "embarked"};
        final String[] classNames = new String[] {"yes", "no"};
        DecisionTree tree = new DecisionTree(nominalAttrs, x, y, numVars, maxDepth, maxLeafs,
            minSplits, minLeafSize, samples, SplitRule.GINI, rand) {
            @Override
            public String toString() {
                return predictJsCodegen(featureNames, classNames);
            }
        };
        tree.toString();
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
