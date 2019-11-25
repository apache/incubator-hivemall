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
package hivemall.ftvec.hashing;

import hivemall.TestUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.hashing.MurmurHash3;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class FeatureHashingUDFTest {

    @Test
    public void testBias() {
        String expected = "0:1.0";
        String actual =
                FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES);
        Assert.assertEquals(expected, actual);

        expected = "0";
        actual = FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES);
        Assert.assertEquals(expected, actual);

        expected = "0:1.1";
        actual = FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES);
        Assert.assertEquals(FeatureHashingUDF.mhash("0", MurmurHash3.DEFAULT_NUM_FEATURES) + ":1.1",
            actual);
    }

    @Test
    public void testBiasLibsvm() {
        String expected = "0:1.0";
        String actual =
                FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES, true);
        Assert.assertEquals(expected, actual);

        expected = "0:1";
        actual = FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES, true);
        Assert.assertEquals(expected, actual);

        expected = "0:1.1";
        actual = FeatureHashingUDF.featureHashing(expected, MurmurHash3.DEFAULT_NUM_FEATURES, true);
        Assert.assertEquals(FeatureHashingUDF.mhash("0", MurmurHash3.DEFAULT_NUM_FEATURES) + ":1.1",
            actual);
    }

    @Test
    public void testEvaluateList() throws HiveException, IOException {
        FeatureHashingUDF udf = new FeatureHashingUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector)});

        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            WritableUtils.val("apple:3", "orange:2", "banana", "0:1"))};

        List<String> expected = Arrays.asList(
            FeatureHashingUDF.mhash("apple", MurmurHash3.DEFAULT_NUM_FEATURES) + ":3",
            FeatureHashingUDF.mhash("orange", MurmurHash3.DEFAULT_NUM_FEATURES) + ":2",
            Integer.toString(FeatureHashingUDF.mhash("banana", MurmurHash3.DEFAULT_NUM_FEATURES)),
            "0:1");
        Assert.assertEquals(expected, udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testEvaluateListLibsvm() throws HiveException, IOException {
        FeatureHashingUDF udf = new FeatureHashingUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector),
                HiveUtils.getConstStringObjectInspector("-libsvm")});

        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            WritableUtils.val("apple:3", "orange:2", "banana", "0:1"))};

        List<String> expected = Arrays.asList(
            FeatureHashingUDF.mhash("apple", MurmurHash3.DEFAULT_NUM_FEATURES) + ":3",
            FeatureHashingUDF.mhash("orange", MurmurHash3.DEFAULT_NUM_FEATURES) + ":2",
            FeatureHashingUDF.mhash("banana", MurmurHash3.DEFAULT_NUM_FEATURES) + ":1", "0:1");
        Collections.sort(expected);
        Assert.assertEquals(expected, udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(FeatureHashingUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-features 1")},
            new Object[] {Arrays.asList("aaa#xxx", "bbb:10")});
    }

}
