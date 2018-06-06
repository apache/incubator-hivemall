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

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class ArrayToStrUDFTest {

    @Test
    public void testSimpleCase() throws HiveException, IOException {
        ArrayToStrUDF udf = new ArrayToStrUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector),
                PrimitiveObjectInspectorFactory.writableStringObjectInspector});

        Text sep = new Text("#");
        DeferredObject[] args =
                new DeferredObject[] {new GenericUDF.DeferredJavaObject(Arrays.asList(1, 2, 3)),
                        new GenericUDF.DeferredJavaObject(sep)};
        Assert.assertEquals("1#2#3", udf.evaluate(args));

        args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(Arrays.asList(1, 2, 3)),
                new GenericUDF.DeferredJavaObject(null)};
        Assert.assertEquals("1,2,3", udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testNoSep() throws HiveException, IOException {
        ArrayToStrUDF udf = new ArrayToStrUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector)});

        DeferredObject[] args =
                new DeferredObject[] {new GenericUDF.DeferredJavaObject(Arrays.asList(1, 2, 3))};

        Assert.assertEquals("1,2,3", udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testNull() throws HiveException, IOException {
        ArrayToStrUDF udf = new ArrayToStrUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector)});

        DeferredObject[] args =
                new DeferredObject[] {new GenericUDF.DeferredJavaObject(Arrays.asList(1, null, 3))};

        Assert.assertEquals("1,3", udf.evaluate(args));

        args = new DeferredObject[] {new GenericUDF.DeferredJavaObject(Arrays.asList(null, 2, 3))};

        Assert.assertEquals("2,3", udf.evaluate(args));

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(ArrayToStrUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector),
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector},
            new Object[] {Arrays.asList(1, 2, 3), "-"});
    }
}
