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
        name = "xgboost_multiclass_predict",
        value = "_FUNC_(string rowid, string[] features, string model_id, array<byte> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, int label, float probability)")
public final class XGBoostMulticlassPredictUDTF extends hivemall.xgboost.XGBoostPredictUDTF {

    public XGBoostMulticlassPredictUDTF() {
        super();
    }

    /** Return (string rowid, int label, float probability) as a result */
    @Override
    protected StructObjectInspector getReturnOI() {
        final List<String> fieldNames = new ArrayList<>(3);
        final List<ObjectInspector> fieldOIs = new ArrayList<>(3);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("label");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("probability");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected void forwardPredicted(@Nonnull final List<LabeledPointWithRowId> testData,
            @Nonnull final float[][] predicted) throws HiveException {
        Preconditions.checkArgument(predicted.length == testData.size(), HiveException.class);

        final Object[] forwardObj = new Object[3];
        for (int i = 0, size = testData.size(); i < size; i++) {
            final float[] predicted_i = predicted[i];
            final String rowId = testData.get(i).getRowId();
            forwardObj[0] = rowId;

            assert (predicted_i.length > 1);
            for (int j = 0; j < predicted_i.length; j++) {
                forwardObj[1] = j;
                float prob = predicted_i[j];
                forwardObj[2] = prob;
                forward(forwardObj);
            }
        }
    }

}
