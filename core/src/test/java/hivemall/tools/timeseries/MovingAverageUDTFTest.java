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
package hivemall.tools.timeseries;

import hivemall.TestUtils;
import hivemall.tools.timeseries.MovingAverageUDTF;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class MovingAverageUDTFTest {

    @Test
    public void test() throws HiveException {
        MovingAverageUDTF udtf = new MovingAverageUDTF();

        ObjectInspector argOI0 = PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
        ObjectInspector argOI1 = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector, 3);

        final List<Double> results = new ArrayList<>();
        udtf.initialize(new ObjectInspector[] {argOI0, argOI1});
        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                Object[] objs = (Object[]) input;
                Assert.assertEquals(1, objs.length);
                Assert.assertTrue(objs[0] instanceof DoubleWritable);
                double x = ((DoubleWritable) objs[0]).get();
                results.add(x);
            }
        });

        udtf.process(new Object[] {1.f, null});
        udtf.process(new Object[] {2.f, null});
        udtf.process(new Object[] {3.f, null});
        udtf.process(new Object[] {4.f, null});
        udtf.process(new Object[] {5.f, null});
        udtf.process(new Object[] {6.f, null});
        udtf.process(new Object[] {7.f, null});

        Assert.assertEquals(Arrays.asList(1.d, 1.5d, 2.d, 3.d, 4.d, 5.d, 6.d), results);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(MovingAverageUDTF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector, 3)},
            new Object[][] {{1.f}, {2.f}, {3.f}, {4.f}, {5.f}});
    }

}
