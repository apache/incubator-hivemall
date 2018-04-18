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
package hivemall.knn.distance;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import hivemall.TestUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class EuclidDistanceUDFTest {

    @Test
    public void test1() {
        List<String> ftvec1 = Arrays.asList("1:1.0", "2:2.0", "3:3.0");
        List<String> ftvec2 = Arrays.asList("1:2.0", "2:4.0", "3:6.0");
        double d = EuclidDistanceUDF.euclidDistance(ftvec1, ftvec2);
        Assert.assertEquals(Math.sqrt(1.0 + 4.0 + 9.0), d, 0.f);
    }

    @Test
    public void test2() {
        List<String> ftvec1 = Arrays.asList("1:1.0", "2:3.0", "3:3.0");
        List<String> ftvec2 = Arrays.asList("1:2.0", "3:6.0");
        double d = EuclidDistanceUDF.euclidDistance(ftvec1, ftvec2);
        Assert.assertEquals(Math.sqrt(1.0 + 9.0 + 9.0), d, 0.f);
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(
            EuclidDistanceUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector)},
            new Object[] {Arrays.asList("1:1.0", "2:3.0", "3:3.0"), Arrays.asList("1:2.0", "3:6.0")});
    }

}
