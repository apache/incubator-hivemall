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
package hivemall.ftvec.trans;

import hivemall.TestUtils;
import hivemall.utils.hadoop.WritableUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;

import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class QuantifiedFeaturesUDTFTest {

    @Test
    public void testSerialization() throws HiveException {
        final QuantifiedFeaturesUDTF udtf = new QuantifiedFeaturesUDTF();

        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true),
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        final List<Object[]> rows = new ArrayList<>();
        udtf.setCollector(new Collector() {
            public void collect(Object input) throws HiveException {
                rows.add((Object[]) input);
            }
        });

        udtf.process(new Object[] {WritableUtils.val(true), "aaa", 1.0});
        udtf.process(new Object[] {WritableUtils.val(true), "bbb", 2.0});

        // test Kryo serialization
        byte[] serialized = TestUtils.serializeObjectByKryo(udtf);
        TestUtils.deserializeObjectByKryo(serialized, QuantifiedFeaturesUDTF.class);

        udtf.close();

        Assert.assertEquals(2, rows.size());

        List<DoubleWritable> features = (List<DoubleWritable>) rows.get(0)[0];
        Assert.assertTrue(features.get(0).get() == 0.d);
        Assert.assertTrue(features.get(1).get() == 1.d);

        features = (List<DoubleWritable>) rows.get(1)[0];
        Assert.assertTrue(features.get(0).get() == 1.d);
        Assert.assertTrue(features.get(1).get() == 2.d);
    }
}
