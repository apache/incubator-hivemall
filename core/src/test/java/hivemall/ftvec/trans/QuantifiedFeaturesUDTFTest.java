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
    public void test() throws HiveException {
        final QuantifiedFeaturesUDTF udtf = new QuantifiedFeaturesUDTF();

        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true),
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        final List<List<Double>> quantifiedInputs = new ArrayList<>();
        udtf.setCollector(new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] row = (Object[]) input;
                @SuppressWarnings("unchecked")
                List<DoubleWritable> column = (List<DoubleWritable>) row[0];
                List<Double> quantifiedInput = new ArrayList<>();
                for (DoubleWritable elem : column) {
                    quantifiedInput.add(elem.get());
                }
                quantifiedInputs.add(quantifiedInput);
            }
        });

        udtf.process(new Object[] {WritableUtils.val(true), "aaa", 1.0});
        udtf.process(new Object[] {WritableUtils.val(true), "bbb", 2.0});

        udtf.close();

        Assert.assertEquals(2, quantifiedInputs.size());

        List<Double> quantifiedInput = quantifiedInputs.get(0);
        Assert.assertTrue(quantifiedInput.get(0) == 0.d);
        Assert.assertTrue(quantifiedInput.get(1) == 1.d);

        quantifiedInput = quantifiedInputs.get(1);
        Assert.assertTrue(quantifiedInput.get(0) == 1.d);
        Assert.assertTrue(quantifiedInput.get(1) == 2.d);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(QuantifiedFeaturesUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true),
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector},
            new Object[][] {{WritableUtils.val(true), "aaa", 1.0}});
    }
}
