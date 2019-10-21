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
package hivemall.ftvec.trans;

import hivemall.TestUtils;
import hivemall.utils.hadoop.WritableUtils;

import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Test;

public class BinarizeLabelUDTFTest {

    // ignored to avoid
    // org.apache.hadoop.hive.shims.ShimLoader.getMajorVersion(ShimLoader.java:141) ExceptionInInitializerError
    // in Hive v0.13.0
    //@Test(expected = UDFArgumentException.class)
    public void testInsufficientLabelColumn() throws HiveException {
        BinarizeLabelUDTF udtf = new BinarizeLabelUDTF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        argOIs[0] = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        List<String> featureNames = Arrays.asList("positive", "features");
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, featureNames);

        udtf.initialize(argOIs);
    }

    @Test
    public void testSerialization() throws HiveException {
        final List<String> featureNames = Arrays.asList("positive", "negative", "features");
        TestUtils.testGenericUDTFSerialization(BinarizeLabelUDTF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorFactory.getStandardConstantListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector, featureNames)},
            new Object[][] {{new Integer(0), new Integer(0), WritableUtils.val("a:1", "b:2")}});
    }
}
