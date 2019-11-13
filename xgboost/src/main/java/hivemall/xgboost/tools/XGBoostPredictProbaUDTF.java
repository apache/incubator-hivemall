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

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "xgboost_predict_proba",
        value = "_FUNC_(string rowid, string[] features, string model_id, array<string> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, double predicted)")
public final class XGBoostPredictProbaUDTF extends hivemall.xgboost.XGBoostOnlinePredictUDTF {

    public XGBoostPredictProbaUDTF() {
        super(new Object[2]);
    }

    /** Return (string rowid, double predicted) as a result */
    @Override
    protected StructObjectInspector getReturnOI() {
        List<String> fieldNames = new ArrayList<>(2);
        List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("predicted");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected void forwardPredicted(@Nonnull String rowId, @Nonnull double[] predicted)
            throws HiveException {
        final Object[] forwardObj = _forwardObj;
        forwardObj[0] = rowId;
        forwardObj[1] = Double.valueOf(predicted[0]);
        forward(forwardObj);
    }

}
