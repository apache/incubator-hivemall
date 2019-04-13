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
package hivemall.regression;

import hivemall.TestUtils;

import java.util.Arrays;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Test;

public class PassiveAggressiveRegressionUDTFTest {

    @Test
    public void testPA1Serialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(PassiveAggressiveRegressionUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector},
            new Object[][] {{Arrays.asList("1:-2", "2:-1"), 1.f}});
    }

    @Test
    public void testPA1() throws HiveException {
        PassiveAggressiveRegressionUDTF udtf = new PassiveAggressiveRegressionUDTF();
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaFloatObjectInspector});
        udtf.setCollector(new Collector() {
            public void collect(Object input) throws HiveException {
                // noop
            }
        });

        udtf.process(new Object[] {Arrays.asList("1:-2", "2:-1"), 1.1f});
        udtf.process(new Object[] {Arrays.asList("3:-2", "1:-1"), -1.3f});

        byte[] serialized = TestUtils.serializeObjectByKryo(udtf);
        TestUtils.deserializeObjectByKryo(serialized, PassiveAggressiveRegressionUDTF.class);

        udtf.close();
    }


}
