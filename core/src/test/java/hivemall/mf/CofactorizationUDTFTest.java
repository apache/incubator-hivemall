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

//    @Test
//    public void testTrain() throws HiveException, IOException {
//        initialize(true, "-max_iters 5 -factors 100 -c0 0.03 -c1 0.3");
//
//        TrainingSample trainSample = new TrainingSample();
//
//        BufferedReader train = readFile("ml100k-cofactor.trainval.gz");
//        String line;
//        while ((line = train.readLine()) != null) {
//            parseLine(line, trainSample);
//            udtf.process(trainSample.toArray());
//        }
//        Assert.assertEquals(udtf.numTraining, 52287);
//        Assert.assertEquals(udtf.numValidations, 9227);
//        udtf.close();
//    }

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
        Assert.assertTrue(cols.length == 4 || cols.length == 5);
        sample.context = cols[0];
        boolean isItem = Integer.parseInt(cols[1]) == 1;
        sample.isValidation = Integer.parseInt(cols[2]) == 1;
        sample.features = parseFeatures(cols[3]);
        sample.sppmi = cols.length == 5 ? parseFeatures(cols[4]) : null;
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
