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

public class ArrayFlattenUDFTest {

    @Test
    public void testEvaluate() throws HiveException, IOException {
        ArrayFlattenUDF udf = new ArrayFlattenUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaIntObjectInspector))});

        DeferredObject[] args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            Arrays.asList(Arrays.asList(0, 1, 2, 3), Arrays.asList(4, 5), Arrays.asList(6, 7)))};

        List<Object> result = udf.evaluate(args);

        Assert.assertEquals(8, result.size());
        for (int i = 0; i < 8; i++) {
            Assert.assertEquals(new IntWritable(i), result.get(i));
        }

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ArrayFlattenUDF.class,
            new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector))},
            new Object[] {Arrays.asList(Arrays.asList(0, 1, 2, 3), Arrays.asList(4, 5),
                Arrays.asList(6, 7))});
    }

}
