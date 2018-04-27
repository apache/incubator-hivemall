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
package hivemall.geospatial;

import java.io.IOException;

import hivemall.TestUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.junit.Assert;
import org.junit.Test;

public class Lat2TileYUDFTest {

    @Test
    public void testEvaluate() throws HiveException, IOException {
        Lat2TileYUDF udf = new Lat2TileYUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector});

        IntWritable result1 = udf.evaluate(
            new DeferredObject[] {new DeferredJavaObject(49.60055d), new DeferredJavaObject(13)});
        Assert.assertEquals(2792, result1.get());

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(Lat2TileYUDF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector},
            new Object[] {49.60055d, 13});
    }

}
