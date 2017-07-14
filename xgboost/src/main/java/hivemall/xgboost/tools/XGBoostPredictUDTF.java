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
package hivemall.xgboost.tools;

import hivemall.utils.lang.Preconditions;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(
        name = "xgboost_predict",
        value = "_FUNC_(string rowid, string[] features, string model_id, array<byte> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, float predicted)")
public final class XGBoostPredictUDTF extends hivemall.xgboost.XGBoostPredictUDTF {

    public XGBoostPredictUDTF() {
        super();
    }

    /** Return (string rowid, float predicted) as a result */
    @Override
    protected StructObjectInspector getReturnOI() {
        final List<String> fieldNames = new ArrayList<>(2);
        final List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("predicted");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected void forwardPredicted(@Nonnull final List<LabeledPointWithRowId> testData,
            @Nonnull final float[][] predicted) throws HiveException {
        Preconditions.checkArgument(predicted.length == testData.size(), HiveException.class);

        final Object[] forwardObj = new Object[2];
        for (int i = 0, size = testData.size(); i < size; i++) {
            assert (predicted[i].length == 1);

            final String rowId = testData.get(i).getRowId();
            float p = predicted[i][0];
            forwardObj[0] = rowId;
            forwardObj[1] = p;

            forward(forwardObj);
        }
    }

}
