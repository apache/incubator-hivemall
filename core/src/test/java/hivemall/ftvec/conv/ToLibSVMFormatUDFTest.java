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
package hivemall.ftvec.conv;

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class ToLibSVMFormatUDFTest {

    @Test
    public void testFeatureOnly() throws IOException, HiveException {
        ToLibSVMFormatUDF udf = new ToLibSVMFormatUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-features 10")});

        Assert.assertEquals("3:2.1 7:3.4", udf.evaluate(new DeferredObject[] {
                new DeferredJavaObject(Arrays.asList("apple:3.4", "orange:2.1"))}));

        Assert.assertEquals("3:2.1 7:3.4", udf.evaluate(
            new DeferredObject[] {new DeferredJavaObject(Arrays.asList("7:3.4", "3:2.1"))}));

        udf.close();
    }

    @Test
    public void testFeatureAndIntLabel() throws IOException, HiveException {
        ToLibSVMFormatUDF udf = new ToLibSVMFormatUDF();

        udf.initialize(
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-features 10")});

        Assert.assertEquals("5 3:2.1 7:3.4",
            udf.evaluate(new DeferredObject[] {
                    new DeferredJavaObject(Arrays.asList("apple:3.4", "orange:2.1")),
                    new DeferredJavaObject(5)}));

        udf.close();
    }

    @Test
    public void testFeatureAndFloatLabel() throws IOException, HiveException {
        ToLibSVMFormatUDF udf = new ToLibSVMFormatUDF();

        udf.initialize(
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-features 10")});

        Assert.assertEquals("5.0 3:2.1 7:3.4",
            udf.evaluate(
                new DeferredObject[] {new DeferredJavaObject(Arrays.asList("7:3.4", "3:2.1")),
                        new DeferredJavaObject(5.f)}));

        udf.close();
    }



}
