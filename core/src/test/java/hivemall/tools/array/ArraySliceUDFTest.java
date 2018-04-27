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

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.junit.Assert;
import org.junit.Test;

public class ArraySliceUDFTest {

    @Test
    public void testNonNullReturn() throws IOException, HiveException {
        ArraySliceUDF udf = new ArraySliceUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                PrimitiveObjectInspectorFactory.writableIntObjectInspector});

        IntWritable offset = new IntWritable();
        IntWritable length = new IntWritable();
        DeferredObject arg1 = new GenericUDF.DeferredJavaObject(offset);
        DeferredObject arg2 = new GenericUDF.DeferredJavaObject(length);
        DeferredObject nullarg = new GenericUDF.DeferredJavaObject(null);

        DeferredObject[] args =
                new DeferredObject[] {
                        new GenericUDF.DeferredJavaObject(Arrays.asList("zero", "one", "two",
                            "three", "four", "five", "six", "seven", "eight", "nine", "ten")),
                        arg1, arg2};

        offset.set(0);
        length.set(3);
        List<Object> actual = udf.evaluate(args);
        Assert.assertEquals(Arrays.asList("zero", "one", "two"), actual);

        offset.set(1);
        length.set(-2);
        actual = udf.evaluate(args);
        Assert.assertEquals(
            Arrays.asList("one", "two", "three", "four", "five", "six", "seven", "eight"), actual);

        offset.set(1);
        length.set(0);
        actual = udf.evaluate(args);
        Assert.assertEquals(Collections.emptyList(), actual);

        offset.set(-1);
        length.set(0);
        actual = udf.evaluate(args);
        Assert.assertEquals(Collections.emptyList(), actual);

        offset.set(6);
        args[2] = nullarg;
        actual = udf.evaluate(args);
        Assert.assertEquals(Arrays.asList("six", "seven", "eight", "nine", "ten"), actual);

        udf.close();
    }

    @Test
    public void testNullReturn() throws IOException, HiveException {
        ArraySliceUDF udf = new ArraySliceUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                PrimitiveObjectInspectorFactory.writableIntObjectInspector});

        IntWritable offset = new IntWritable();
        IntWritable length = new IntWritable();
        DeferredObject arg1 = new GenericUDF.DeferredJavaObject(offset);
        DeferredObject arg2 = new GenericUDF.DeferredJavaObject(length);

        DeferredObject[] args =
                new DeferredObject[] {
                        new GenericUDF.DeferredJavaObject(Arrays.asList("zero", "one", "two",
                            "three", "four", "five", "six", "seven", "eight", "nine", "ten")),
                        arg1, arg2};


        offset.set(-12);
        length.set(0);
        List<Object> actual = udf.evaluate(args);
        Assert.assertNull(actual);

        udf.close();

    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ArraySliceUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector},
            new Object[] {Arrays.asList("zero", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten"), 2, 5});
    }

}
