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

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class FromJsonUDFTest {

    @Test
    public void testDoubleArray() throws Exception {
        FromJsonUDF udf = new FromJsonUDF();

        String json = "[0.1,1.1,2.2]";
        String types = "array<double>";
        List<Double> expected = Arrays.asList(0.1d, 1.1d, 2.2d);

        ObjectInspector[] argOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                HiveUtils.getConstStringObjectInspector(types)};
        DeferredObject[] args =
                new DeferredObject[] {new GenericUDF.DeferredJavaObject(new Text(json)), null};

        udf.initialize(argOIs);
        Object result = udf.evaluate(args);

        Assert.assertEquals(expected, result);

        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testPersonStruct() throws Exception {
        FromJsonUDF udf = new FromJsonUDF();

        String json = "{ \"person\" : { \"name\" : \"makoto\" , \"age\" : 37 } }";
        String types = "struct<name:string,age:int>";

        ObjectInspector[] argOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.writableStringObjectInspector,
                HiveUtils.getConstStringObjectInspector(types),
                HiveUtils.getConstStringObjectInspector("person")};
        DeferredObject[] args =
                new DeferredObject[] {new GenericUDF.DeferredJavaObject(new Text(json)), null};

        udf.initialize(argOIs);
        List<Object> result = (List<Object>) udf.evaluate(args);

        Assert.assertEquals(2, result.size());
        Assert.assertEquals("makoto", result.get(0));
        Assert.assertEquals(37, result.get(1));

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(FromJsonUDF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    HiveUtils.getConstStringObjectInspector("array<double>")},
            new Object[] {"[0.1,1.1,2.2]"});
    }

    @Test
    public void testSerializationThreeArgs() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(FromJsonUDF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    HiveUtils.getConstStringObjectInspector("struct<name:string,age:int>"),
                    HiveUtils.getConstStringObjectInspector("person")},
            new Object[] {"{ \"person\" : { \"name\" : \"makoto\" , \"age\" : 37 } }"});
    }

}
