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
package hivemall.xgboost.classification;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class XGBoostMulticlassClassifierUDTFTest {

    @Test
    public void testCheckTargetValueSucess() throws HiveException {
        XGBoostMulticlassClassifierUDTF udtf = new XGBoostMulticlassClassifierUDTF();
        udtf.initialize(
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-num_class 4")});

        udtf.processTargetValue(1.0f);
        udtf.processTargetValue(3f);
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue1() throws HiveException {
        XGBoostMulticlassClassifierUDTF udtf = new XGBoostMulticlassClassifierUDTF();

        udtf.processTargetValue(1.1f);
        Assert.fail();
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue2() throws HiveException {
        XGBoostMulticlassClassifierUDTF udtf = new XGBoostMulticlassClassifierUDTF();
        udtf.processOptions(
            new ObjectInspector[] {null, null, ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-num_class 3")});

        udtf.processTargetValue(-2f);
        Assert.fail();
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue3() throws HiveException {
        XGBoostMulticlassClassifierUDTF udtf = new XGBoostMulticlassClassifierUDTF();
        udtf.processOptions(
            new ObjectInspector[] {null, null, ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-num_class 3")});

        udtf.processTargetValue(3f);
        Assert.fail();
    }

}
