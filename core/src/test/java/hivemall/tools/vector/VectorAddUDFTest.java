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
package hivemall.tools.vector;

import hivemall.TestUtils;
import hivemall.utils.hadoop.WritableUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class VectorAddUDFTest {

    @Test
    public void testAddDouble() throws IOException, HiveException {
        VectorAddUDF udf = new VectorAddUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableFloatObjectInspector)});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {1, 2, 3})),
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new float[] {2, 3, 4}))};

        List<?> actual = udf.evaluate(args);
        List<Double> expected = Arrays.asList(3.d, 5.d, 7.d);

        Assert.assertEquals(expected, actual);

        udf.close();
    }

    @Test
    public void testAddLong() throws IOException, HiveException {
        VectorAddUDF udf = new VectorAddUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableLongObjectInspector),
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector)});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new long[] {1, 2, 3})),
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new int[] {2, 3, 4}))};

        List<?> actual = udf.evaluate(args);
        List<Long> expected = Arrays.asList(3L, 5L, 7L);

        Assert.assertEquals(expected, actual);

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(VectorAddUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaFloatObjectInspector)},
            new Object[] {Arrays.asList(1.d, 2.d, 3.d), Arrays.asList(2.f, 3.f, 4.f)});
    }

}
