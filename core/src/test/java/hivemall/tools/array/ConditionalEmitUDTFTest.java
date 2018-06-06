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
package hivemall.tools.array;

import hivemall.TestUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class ConditionalEmitUDTFTest {

    @Test
    public void test() throws HiveException {
        ConditionalEmitUDTF udtf = new ConditionalEmitUDTF();

        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaBooleanObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),});

        final List<Object> actual = new ArrayList<>();
        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                Object[] forwardObj = (Object[]) input;
                Assert.assertEquals(1, forwardObj.length);
                actual.add(forwardObj[0]);
            }
        });

        udtf.process(
            new Object[] {Arrays.asList(true, false, true), Arrays.asList("one", "two", "three")});

        Assert.assertEquals(Arrays.asList("one", "three"), actual);

        actual.clear();

        udtf.process(
            new Object[] {Arrays.asList(true, true, false), Arrays.asList("one", "two", "three")});
        Assert.assertEquals(Arrays.asList("one", "two"), actual);

        udtf.close();
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(ConditionalEmitUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector),
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector)},
            new Object[][] {
                    {Arrays.asList(true, false, true), Arrays.asList("one", "two", "three")},
                    {Arrays.asList(true, true, false), Arrays.asList("one", "two", "three")}});
    }

}
