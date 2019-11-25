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
package hivemall.factorization.fm;

import hivemall.factorization.fm.Feature;
import hivemall.factorization.fm.IntFeature;
import hivemall.factorization.fm.StringFeature;
import hivemall.utils.hashing.MurmurHash3;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Test;

public class FeatureTest {

    @Test
    public void testParseFeature() throws HiveException {
        Feature f2 = Feature.parseFeature("1164:0.3652", false);
        Assert.assertTrue(f2 instanceof StringFeature);
        Assert.assertEquals("1164", f2.getFeature());
        Assert.assertEquals(0.3652d, f2.getValue(), 0.d);
    }

    @Test
    public void testParseFFMFeature() throws HiveException {
        IntFeature f1 = Feature.parseFFMFeature("2:1163:0.3651", -1);
        Assert.assertEquals(2, f1.getField());
        Assert.assertEquals(1163, f1.getFeatureIndex());
        Assert.assertEquals("1163", f1.getFeature());
        Assert.assertEquals(0.3651d, f1.getValue(), 0.d);
    }

    @Test(expected = HiveException.class)
    public void testParseQuantitativeFFMFeatureFails1() throws HiveException {
        Feature.parseFFMFeature("163:0.3651");
    }

    @Test(expected = HiveException.class)
    public void testParseQuantitativeFFMFeatureFails2() throws HiveException {
        Feature.parseFFMFeature("1163:0.3651");
    }

    @Test
    public void testParseFeatureProbe() throws HiveException {
        Feature probe = Feature.parseFeature("dummy:-1", false);
        Feature.parseFeature("1164:0.3652", probe, false);
        Assert.assertEquals("1164", probe.getFeature());
        Assert.assertEquals(0.3652d, probe.getValue(), 0.d);
    }

    @Test
    public void testParseFFMFeatureProbe() throws HiveException {
        IntFeature probe = Feature.parseFFMFeature("dummyField:dummyFeature:-1");
        Assert.assertEquals(MurmurHash3.murmurhash3("dummyFeature", Feature.DEFAULT_NUM_FEATURES)
                + Feature.DEFAULT_NUM_FIELDS,
            probe.getFeatureIndex());
        Feature.parseFFMFeature("2:1163:0.3651", probe, -1, Feature.DEFAULT_NUM_FIELDS);
        Assert.assertEquals(2, probe.getField());
        Assert.assertEquals(1163, probe.getFeatureIndex());
        Assert.assertEquals("1163", probe.getFeature());
        Assert.assertEquals(0.3651d, probe.getValue(), 0.d);
    }

    @Test
    public void testParseIntFeature() throws HiveException {
        Feature f = Feature.parseFeature("1163:0.3651", true);
        Assert.assertTrue(f instanceof IntFeature);
        Assert.assertEquals("1163", f.getFeature());
        Assert.assertEquals(1163, f.getFeatureIndex());
        Assert.assertEquals(0.3651d, f.getValue(), 0.d);
    }

    @Test(expected = HiveException.class)
    public void testParseIntFeatureFails() throws HiveException {
        Feature.parseFeature("2:1163:0.3651", true);
    }

    @Test(expected = HiveException.class)
    public void testParseFeatureZeroIndex() throws HiveException {
        Feature.parseFFMFeature("0:0.3652");
    }

    @Test
    public void testFFMFeatureL2Normalization() throws HiveException {
        Feature[] features = new Feature[9];
        // (0, 0, 1, 1, 0, 1, 0, 1, 0)
        features[0] = Feature.parseFFMFeature("11:1:0", -1);
        features[1] = Feature.parseFFMFeature("22:2:0", -1);
        features[2] = Feature.parseFFMFeature("33:3:1", -1);
        features[3] = Feature.parseFFMFeature("44:4:1", -1);
        features[4] = Feature.parseFFMFeature("55:5:0", -1);
        features[5] = Feature.parseFFMFeature("66:6:1", -1);
        features[6] = Feature.parseFFMFeature("77:7:0", -1);
        features[7] = Feature.parseFFMFeature("88:8:1", -1);
        features[8] = Feature.parseFFMFeature("99:9:0", -1);
        Assert.assertEquals(features[0].getField(), 11);
        Assert.assertEquals(features[1].getField(), 22);
        Assert.assertEquals(features[2].getField(), 33);
        Assert.assertEquals(features[3].getField(), 44);
        Assert.assertEquals(features[4].getField(), 55);
        Assert.assertEquals(features[5].getField(), 66);
        Assert.assertEquals(features[6].getField(), 77);
        Assert.assertEquals(features[7].getField(), 88);
        Assert.assertEquals(features[8].getField(), 99);
        Assert.assertEquals(features[0].getFeatureIndex(), 1);
        Assert.assertEquals(features[1].getFeatureIndex(), 2);
        Assert.assertEquals(features[2].getFeatureIndex(), 3);
        Assert.assertEquals(features[3].getFeatureIndex(), 4);
        Assert.assertEquals(features[4].getFeatureIndex(), 5);
        Assert.assertEquals(features[5].getFeatureIndex(), 6);
        Assert.assertEquals(features[6].getFeatureIndex(), 7);
        Assert.assertEquals(features[7].getFeatureIndex(), 8);
        Assert.assertEquals(features[8].getFeatureIndex(), 9);
        Assert.assertEquals(0.d, features[0].value, 1E-15);
        Assert.assertEquals(0.d, features[1].value, 1E-15);
        Assert.assertEquals(1.d, features[2].value, 1E-15);
        Assert.assertEquals(1.d, features[3].value, 1E-15);
        Assert.assertEquals(0.d, features[4].value, 1E-15);
        Assert.assertEquals(1.d, features[5].value, 1E-15);
        Assert.assertEquals(0.d, features[6].value, 1E-15);
        Assert.assertEquals(1.d, features[7].value, 1E-15);
        Assert.assertEquals(0.d, features[8].value, 1E-15);
        Feature.l2normalize(features);
        // (0, 0, 0.5, 0.5, 0, 0.5, 0, 0.5, 0)
        Assert.assertEquals(0.d, features[0].value, 1E-15);
        Assert.assertEquals(0.d, features[1].value, 1E-15);
        Assert.assertEquals(0.5d, features[2].value, 1E-15);
        Assert.assertEquals(0.5d, features[3].value, 1E-15);
        Assert.assertEquals(0.d, features[4].value, 1E-15);
        Assert.assertEquals(0.5d, features[5].value, 1E-15);
        Assert.assertEquals(0.d, features[6].value, 1E-15);
        Assert.assertEquals(0.5d, features[7].value, 1E-15);
        Assert.assertEquals(0.d, features[8].value, 1E-15);
    }

}
