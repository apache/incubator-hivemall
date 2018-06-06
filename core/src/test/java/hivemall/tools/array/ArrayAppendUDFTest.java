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
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class ArrayAppendUDFTest {

    @Test
    public void testEvaluate() throws HiveException, IOException {
        ArrayAppendUDF udf = new ArrayAppendUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {0, 1, 2})),
                new GenericUDF.DeferredJavaObject(new Double(3))};

        List<Object> result = udf.evaluate(args);

        Assert.assertEquals(4, result.size());
        for (int i = 0; i < 4; i++) {
            Assert.assertEquals(new DoubleWritable(i), result.get(i));
        }

        udf.close();
    }

    @Test
    public void testEvaluateAvoidNullAppend() throws HiveException, IOException {
        ArrayAppendUDF udf = new ArrayAppendUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(
                    WritableUtils.toWritableList(new double[] {0, 1, 2})),
                new GenericUDF.DeferredJavaObject(null)};

        List<Object> result = udf.evaluate(args);

        Assert.assertEquals(3, result.size());
        for (int i = 0; i < 3; i++) {
            Assert.assertEquals(new DoubleWritable(i), result.get(i));
        }

        udf.close();
    }

    @Test
    public void testEvaluateNullList() throws HiveException, IOException {
        ArrayAppendUDF udf = new ArrayAppendUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(null),
                new GenericUDF.DeferredJavaObject(new Double(3d))};

        List<Object> result = udf.evaluate(args);

        Assert.assertEquals(Arrays.asList(new DoubleWritable(3d)), result);

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ArrayAppendUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector},
            new Object[] {Arrays.asList(0.d, 1.d, 2.d), 3.d});
    }

}
