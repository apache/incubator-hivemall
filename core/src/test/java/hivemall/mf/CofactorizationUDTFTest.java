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
package hivemall.mf;

import hivemall.fm.Feature;
import hivemall.fm.StringFeature;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.MapredContextAccessor;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class CofactorizationUDTFTest {

    CofactorizationUDTF udtf;

    private static class TrainingSample {
        private String context;
        private List<String> features;
        private List<String> sppmi;
        private boolean isValidation;

        private TrainingSample() {}

        private Object[] toArray() {
            boolean isItem = sppmi != null;
            return new Object[]{context, features, isValidation, isItem, sppmi};
        }
    }

    private static class TestingSample {
        private String user;
        private String item;
        private double rating;

        private TestingSample() {}
    }

    @Before
    public void setUp() throws HiveException {
        udtf = new CofactorizationUDTF();
    }

    private void initialize(final boolean initMapred, @Nonnull final String options) throws HiveException {
        if (initMapred) {
            MapredContext mapredContext = MapredContextAccessor.create(true, null);
            udtf.configure(mapredContext);
            udtf.setCollector(new Collector() {
                @Override
                public void collect(Object args) throws HiveException {
                }
            });
        }

        ObjectInspector[] argOIs = new ObjectInspector[]{
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaBooleanObjectInspector,
                PrimitiveObjectInspectorFactory.javaBooleanObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                HiveUtils.getConstStringObjectInspector(options)
        };
        udtf.initialize(argOIs);
    }

    @Test
    public void testCreateNnzFeatureArray() throws HiveException {
        Feature item1 = new StringFeature("item1", 0);
        Feature item2 = new StringFeature("item2", 1);
        Feature item3 = new StringFeature("item3", 0);

        Feature[] expected = new Feature[]{item2};

        Feature[] actual = CofactorizationUDTF.createNnzFeatureArray(new Feature[]{
                item1, item2, item3
        });

        Assert.assertArrayEquals(actual, expected);
    }

    @Test
    public void testTrain() throws HiveException, IOException {
        initialize(true, "-max_iters 1 -factors 2 -c0 0.03 -c1 0.3");

        TrainingSample trainSample = new TrainingSample();

        BufferedReader train = readFile("ml100k-cofactor.train.gz");
        String line;
        while ((line = train.readLine()) != null) {
            parseLine(line, trainSample);
            udtf.process(trainSample.toArray());
        }
        CofactorModel model = udtf.model;
        udtf.close();

        TestingSample trainTestSample = new TestingSample();
        BufferedReader trainTest = readFile("ml100k-cofactor.train.repeat.gz");
        double err = 0.d;
        int numTrainTest = 0;

        while ((line = trainTest.readLine()) != null) {
            numTrainTest++;
            parseLine(line, trainTestSample);
            Double prediction = model.predict(trainTestSample.user, trainTestSample.item);
            if (prediction == null) {
                continue;
            }
            err += Math.abs(trainTestSample.rating - prediction);
            System.out.println(trainTestSample.rating + ", " + prediction);
//            Assert.assertTrue(err < Double.MAX_VALUE && err > Double.MIN_VALUE);
            if (numTrainTest == 100) {
                break;
            }
        }
        System.out.println(err / numTrainTest);

        TestingSample testSample = new TestingSample();
        BufferedReader test = readFile("ml100k-cofactor.test.gz");
        err = 0.d;
        int numTest = 0;

        while ((line = test.readLine()) != null) {
            numTest++;
            parseLine(line, testSample);
            double prediction = model.predict(testSample.user, testSample.item);
            err += Math.abs(testSample.rating - prediction);
            System.out.println(testSample.rating + ", " + prediction);
            if (numTest == 100) {
                break;
            }
        }
        System.out.println(err / numTest);
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = CofactorizationUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    private static void parseLine(@Nonnull String line, @Nonnull TrainingSample sample) {
        String[] cols = StringUtils.split(line, ' ');
        Assert.assertNotNull(cols);
        Assert.assertTrue(cols.length == 3 || cols.length == 4);
        sample.context = cols[0];
        boolean isItem = Integer.parseInt(cols[1]) == 1;
        sample.features = parseFeatures(cols[2]);
        sample.sppmi = cols.length == 4 ? parseFeatures(cols[3]) : null;
    }

    private static void parseLine(@Nonnull String line, @Nonnull TestingSample sample) {
        String[] cols = StringUtils.split(line, ' ');
        Assert.assertNotNull(cols);
        Assert.assertEquals(cols.length, 3);
        sample.user = cols[0];
        sample.item = cols[1];
        sample.rating = Double.parseDouble(cols[2]);
    }


    private static List<String> parseFeatures(@Nonnull String string) {
        String[] entries = StringUtils.split(string, ',');
        List<String> features = new ArrayList<>();
        features.addAll(Arrays.asList(entries));
        return features;
    }

    @Test
    public void readMiniBatchFromFile_oneTrainSample_success() throws HiveException {
        initialize(false, "");
        Object[] trainSample = getItemTrainSample();
        udtf.process(trainSample);
        udtf.prepareForRead();

        CofactorizationUDTF.MiniBatch miniBatch = new CofactorizationUDTF.MiniBatch();

        boolean didRead = udtf.readMiniBatchFromFile(miniBatch);
        Assert.assertTrue(didRead);
        Assert.assertEquals(miniBatch.trainingSize(), 1);
        Assert.assertEquals(miniBatch.getItems().size(), 1);

        CofactorizationUDTF.TrainingSample actualSample = miniBatch.getItems().get(0);
        assertSamplesAreEqual(trainSample, actualSample);
    }

    @Test
    public void readMiniBatchFromFile_oneValidSample_success() throws HiveException {
        initialize(false, "");
        Object[] validSample = getItemValidationSample();
        udtf.process(validSample);
        udtf.prepareForRead();

        CofactorizationUDTF.MiniBatch miniBatch = new CofactorizationUDTF.MiniBatch();

        boolean didRead = udtf.readMiniBatchFromFile(miniBatch);
        Assert.assertTrue(didRead);
        Assert.assertEquals(miniBatch.validationSize(), 1);

        CofactorizationUDTF.TrainingSample actualSample = miniBatch.getValidationSamples().get(0);
        assertSamplesAreEqual(validSample, actualSample);
    }

    @Test
    public void readMiniBatchFromFile_twoTrainingSamples_success() throws HiveException {
        initialize(false, "");

        Object[] item = getItemTrainSample();
        Object[] user = getUserSample();

        udtf.process(item);
        udtf.process(user);
        udtf.prepareForRead();

        CofactorizationUDTF.MiniBatch miniBatch = new CofactorizationUDTF.MiniBatch();

        boolean didRead = udtf.readMiniBatchFromFile(miniBatch);
        Assert.assertTrue(didRead);
        Assert.assertEquals(miniBatch.trainingSize(), 2);
        Assert.assertEquals(miniBatch.getItems().size(), 1);
        Assert.assertEquals(miniBatch.getUsers().size(), 1);

        CofactorizationUDTF.TrainingSample actualItem = miniBatch.getItems().get(0);
        assertSamplesAreEqual(item, actualItem);

        CofactorizationUDTF.TrainingSample actualUser = miniBatch.getUsers().get(0);
        assertSamplesAreEqual(user, actualUser);
    }

    @Test
    public void process_fourArgs_success() throws HiveException {
        initialize(false, "");
        udtf.process(new Object[]{"string", getDummyFeatures(), false, true, getDummyFeatures()});
    }

    @Test(expected = HiveException.class)
    public void process_threeArgs_throwsException() throws HiveException {
        initialize(false, "");
        udtf.process(new Object[]{"string", getDummyFeatures(), false, true});
    }

    @Test
    public void process_sampleTypes_success() throws HiveException {
        initialize(false, "");
        udtf.process(new Object[]{"train1", getDummyFeatures(), false, false, null});
        udtf.process(new Object[]{"train2", getDummyFeatures(), false, true, getDummyFeatures()});
        udtf.process(new Object[]{"valid1", getDummyFeatures(), true, true, getDummyFeatures()});
        udtf.process(new Object[]{"valid2", getDummyFeatures(), true, false, null});
        Assert.assertEquals(2, udtf.numTraining);
        Assert.assertEquals(2, udtf.numValidations);
    }

    @Test(expected = HiveException.class)
    public void process_trainingUsersHaveDuplicateContext_throwsException() throws HiveException {
        initialize(false, "");
        udtf.process(new Object[]{"train1", getDummyFeatures(), false, false, null});
        udtf.process(new Object[]{"train1", getDummyFeatures(), false, false, getDummyFeatures()});
    }

    @Test
    public void process_trainingAndValidationUsersHaveDuplicateContext_success() throws HiveException {
        initialize(false, "");
        udtf.process(new Object[]{"nancy", getDummyFeatures(), false, false, null});
        udtf.process(new Object[]{"nancy", getDummyFeatures(), true, false, getDummyFeatures()});
        Assert.assertEquals(1, udtf.numTraining);
        Assert.assertEquals(1, udtf.numValidations);
    }

    private void assertSamplesAreEqual(@Nonnull final Object[] expected, @Nonnull final CofactorizationUDTF.TrainingSample actual) throws HiveException {
        Assert.assertEquals(expected[0], actual.context);
        Assert.assertTrue(featureArraysAreEqual(CofactorizationUDTF.parseFeatures(expected[1], udtf.featuresOI, null), actual.features));
        Assert.assertEquals(expected[2], actual.isValidation);
        Assert.assertEquals(expected[3], actual.isItem());
        Assert.assertTrue(featureArraysAreEqual(CofactorizationUDTF.parseFeatures(expected[4], udtf.sppmiOI, null), actual.sppmi));
    }

    private static boolean featureArraysAreEqual(Feature[] f1, Feature[] f2) {
        if (f1 == null && f2 == null) {
            return true;
        }
        if (f1 == null || f2 == null) {
            return false;
        }
        if (f1.length != f2.length) {
            return false;
        }
        for (int i = 0; i < f1.length; i++) {
            if (!featuresAreEqual(f1[i], f2[i])) {
                return false;
            }
        }
        return true;
    }

    private static boolean featuresAreEqual(Feature f1, Feature f2) {
        return f1.getFeature().equals(f2.getFeature()) && f1.getValue() == f2.getValue();
    }

    private static Object[] getItemTrainSample() {
        return new Object[]{"string1", getDummyFeatures(), false, true, getDummyFeatures()};
    }

    private static Object[] getItemValidationSample() {
        return new Object[]{"string1", getDummyFeatures(), true, true, getDummyFeatures()};
    }

    private static Object[] getUserSample() {
        return new Object[]{"user", getDummyFeatures(), false, false, null};
    }

    private static List<String> getDummyFeatures() {
        List<String> features = new ArrayList<>();
        features.add("feature1:1");
        features.add("feature2:2");
        features.add("feature3:3");
        return features;
    }
}
