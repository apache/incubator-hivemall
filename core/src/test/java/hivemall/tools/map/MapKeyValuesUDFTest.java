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
package hivemall.tools.map;

import hivemall.TestUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class MapKeyValuesUDFTest {


    @Test
    public void testStringDouble() throws HiveException, IOException {
        MapKeyValuesUDF udf = new MapKeyValuesUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)});

        Map<String, DoubleWritable> input = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            input.put("k" + i, new DoubleWritable(i));
        }

        GenericUDF.DeferredObject[] arguments =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(input)};

        List<Object[]> actual = udf.evaluate(arguments);

        Assert.assertEquals(input.size(), actual.size());
        for (Object[] e : actual) {
            Assert.assertEquals(2, e.length);
            Object v = input.get(e[0]);
            Assert.assertEquals(e[1], v);
        }

        udf.close();
    }

    @Test
    public void testSerialization() throws UDFArgumentException {
        MapKeyValuesUDF udf = new MapKeyValuesUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)});

        byte[] serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, MapKeyValuesUDF.class);
    }

}
