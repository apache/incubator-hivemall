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
import hivemall.utils.hadoop.WritableUtils;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class ArrayElementAtUDFTest {

    @Test
    public void testDouble() throws IOException, HiveException {
        ArrayElementAtUDF udf = new ArrayElementAtUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {0, 1, 2})),
                new GenericUDF.DeferredJavaObject(new Integer(1))};

        Assert.assertEquals(new DoubleWritable(1), udf.evaluate(args));

        args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {0, 1, 2})),
                new GenericUDF.DeferredJavaObject(new Integer(4))};
        Assert.assertNull(udf.evaluate(args));

        args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {0, 1, 2})),
                new GenericUDF.DeferredJavaObject(new Integer(-2))};
        Assert.assertEquals(new DoubleWritable(1), udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testString() throws IOException, HiveException {
        ArrayElementAtUDF udf = new ArrayElementAtUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(WritableUtils.val("s0", "s1", "s2")),
                new GenericUDF.DeferredJavaObject(1)};

        Assert.assertEquals(WritableUtils.val("s1"), udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ArrayElementAtUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector},
            new Object[] {Arrays.asList(0.d, 1.d, 2.d), 1});
    }

}
