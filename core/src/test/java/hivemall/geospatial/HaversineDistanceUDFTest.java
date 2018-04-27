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
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class HaversineDistanceUDFTest {

    @Test
    public void testKilometers1() throws HiveException, IOException {
        HaversineDistanceUDF udf = new HaversineDistanceUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        // Tokyo
        double lat1 = 35.6833d, lon1 = 139.7667d;
        // Osaka
        double lat2 = 34.6603d, lon2 = 135.5232d;

        DoubleWritable result1 = udf.evaluate(
            new DeferredObject[] {new DeferredJavaObject(lat1), new DeferredJavaObject(lon1),
                    new DeferredJavaObject(lat2), new DeferredJavaObject(lon2)});
        Assert.assertEquals(402.092d, result1.get(), 0.001d);

        udf.close();
    }

    @Test
    public void testKilometers2() throws HiveException, IOException {
        HaversineDistanceUDF udf = new HaversineDistanceUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, false)});

        // Tokyo
        double lat1 = 35.6833d, lon1 = 139.7667d;
        // Osaka
        double lat2 = 34.6603d, lon2 = 135.5232d;

        DoubleWritable result1 = udf.evaluate(
            new DeferredObject[] {new DeferredJavaObject(lat1), new DeferredJavaObject(lon1),
                    new DeferredJavaObject(lat2), new DeferredJavaObject(lon2)});
        Assert.assertEquals(402.092d, result1.get(), 0.001d);

        udf.close();
    }

    @Test
    public void testMiles() throws HiveException, IOException {
        HaversineDistanceUDF udf = new HaversineDistanceUDF();
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true)});

        // Tokyo
        double lat1 = 35.6833d, lon1 = 139.7667d;
        // Osaka
        double lat2 = 34.6603d, lon2 = 135.5232d;

        DoubleWritable result1 = udf.evaluate(new DeferredObject[] {new DeferredJavaObject(lat1),
                new DeferredJavaObject(lon1), new DeferredJavaObject(lat2),
                new DeferredJavaObject(lon2), new DeferredJavaObject(true)});
        Assert.assertEquals(249.84d, result1.get(), 0.1d);

        udf.close();
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        // Tokyo
        double lat1 = 35.6833d, lon1 = 139.7667d;
        // Osaka
        double lat2 = 34.6603d, lon2 = 135.5232d;

        TestUtils.testGenericUDFSerialization(HaversineDistanceUDF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true)},
            new Object[] {lat1, lon1, lat2, lon2, true});
    }

}
