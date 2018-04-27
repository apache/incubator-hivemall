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
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class FirstElementUDFTest {

    @Test
    public void test() throws IOException, HiveException {
        FirstElementUDF udf = new FirstElementUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)});

        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            WritableUtils.toWritableList(new double[] {0, 1, 2}))};

        Assert.assertEquals(WritableUtils.val(0.d), udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testNull() throws IOException, HiveException {
        FirstElementUDF udf = new FirstElementUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)});

        DeferredObject[] args = new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(WritableUtils.toWritableList(new double[] {}))};

        Assert.assertNull(udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(FirstElementUDF.class,
            new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)},
            new Object[] {Arrays.asList(0.d, 1.d, 2.d)});
    }

}
