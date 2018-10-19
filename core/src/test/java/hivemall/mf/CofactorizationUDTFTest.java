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
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CofactorizationUDTFTest {

    CofactorizationUDTF udtf;

    @Before
    public void setUp() throws HiveException {
        this.udtf = new CofactorizationUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[]{
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaBooleanObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                HiveUtils.getConstStringObjectInspector("-num_items 100 -factors 10")
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
    public void readMiniBatchFromFile_oneSample_success() throws HiveException, IOException {
        Object[] sample = getItemSample();
        udtf.process(sample);
        udtf.prepareForRead();

        CofactorizationUDTF.MiniBatch miniBatch = new CofactorizationUDTF.MiniBatch();

        boolean didRead = udtf.readMiniBatchFromFile(miniBatch);
        Assert.assertTrue(didRead);
        Assert.assertEquals(miniBatch.size(), 1);
        Assert.assertEquals(miniBatch.getItems().size(), 1);

        CofactorizationUDTF.TrainingSample actualSample = miniBatch.getItems().get(0);
        Assert.assertEquals(actualSample.context, sample[0]);
        Assert.assertTrue(Arrays.deepEquals(actualSample.features, CofactorizationUDTF.parseFeatures(sample[1], udtf.featuresOI, null)));
        Assert.assertEquals(actualSample.isItem(), sample[2]);
        Assert.assertTrue(Arrays.deepEquals(actualSample.sppmi, CofactorizationUDTF.parseFeatures(sample[3], udtf.sppmiOI, null)));
    }

    @Test
    public void readMiniBatchFromFile_twoSamples_success() throws HiveException, IOException {
        Object[] item = getItemSample();
        Object[] user = getUserSample();

        udtf.process(item);
        udtf.process(user);
        udtf.prepareForRead();

        CofactorizationUDTF.MiniBatch miniBatch = new CofactorizationUDTF.MiniBatch();

        boolean didRead = udtf.readMiniBatchFromFile(miniBatch);
        Assert.assertTrue(didRead);
        Assert.assertEquals(miniBatch.size(), 2);
        Assert.assertEquals(miniBatch.getItems().size(), 1);
        Assert.assertEquals(miniBatch.getUsers().size(), 1);

        CofactorizationUDTF.TrainingSample actualItem = miniBatch.getItems().get(0);
        Assert.assertEquals(actualItem.context, item[0]);
        Assert.assertTrue(Arrays.deepEquals(actualItem.features, CofactorizationUDTF.parseFeatures(item[1], udtf.featuresOI, null)));
        Assert.assertEquals(actualItem.isItem(), item[2]);
        Assert.assertTrue(Arrays.deepEquals(actualItem.sppmi, CofactorizationUDTF.parseFeatures(item[3], udtf.sppmiOI, null)));

        CofactorizationUDTF.TrainingSample actualUser = miniBatch.getUsers().get(0);
        Assert.assertEquals(actualUser.context, user[0]);
        Assert.assertTrue(Arrays.deepEquals(actualUser.features, CofactorizationUDTF.parseFeatures(user[1], udtf.featuresOI, null)));
        Assert.assertEquals(actualUser.isItem(), user[2]);
        Assert.assertTrue(Arrays.deepEquals(actualUser.sppmi, CofactorizationUDTF.parseFeatures(user[3], udtf.sppmiOI, null)));
    }

    @Test
    public void process_fourArgs_success() throws HiveException {
        udtf.process(new Object[]{"string", getDummyFeatures(), true, getDummyFeatures()});
    }

    @Test(expected = HiveException.class)
    public void process_threeArgs_throwsException() throws HiveException {
        udtf.process(new Object[]{"string", getDummyFeatures(), true});
    }

    private static Object[] getItemSample() {
        return new Object[]{"string1", getDummyFeatures(), true, getDummyFeatures()};
    }

    private static Object[] getUserSample() {
        return new Object[]{"user", getDummyFeatures(), false, null};
    }

    private static List<String> getDummyFeatures() {
        List<String> features = new ArrayList<>();
        features.add("feature1:1");
        features.add("feature2:2");
        features.add("feature3:3");
        return features;
    }
}
