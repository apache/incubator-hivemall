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
package hivemall.tools.json;

import hivemall.TestUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class ToJsonUDFTest {

    @Test
    public void testDoubleArray() throws Exception {
        ToJsonUDF udf = new ToJsonUDF();

        ObjectInspector[] argOIs =
                new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)};
        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            WritableUtils.toWritableList(new double[] {0.1, 1.1, 2.1}))};

        udf.initialize(argOIs);
        Text serialized = udf.evaluate(args);

        Assert.assertEquals("[0.1,1.1,2.1]", serialized.toString());

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ToJsonUDF.class,
            new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)},
            new Object[] {Arrays.asList(0.1d, 1.1d, 2.1d)});
    }

    @Test
    public void testSerializationTwoArgs() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ToJsonUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                    HiveUtils.getConstStringObjectInspector("person")},
            new Object[] {Arrays.asList(0.1d, 1.1d, 2.1d)});
    }

}
