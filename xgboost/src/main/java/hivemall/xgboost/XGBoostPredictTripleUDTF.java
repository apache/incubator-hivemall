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
package hivemall.xgboost;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Writable;

//@formatter:off
@Description(name = "xgboost_predict_triple",
        value = "_FUNC_(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, string label, double probability)",
        extended = "select\n" + 
                "  rowid,\n" + 
                "  label,\n" + 
                "  avg(prob) as prob\n" + 
                "from (\n" + 
                "  select\n" + 
                "    xgboost_predict_triple(rowid, features, model_id, model) as (rowid, label, prob)\n" + 
                "  from\n" + 
                "    xgb_model l\n" + 
                "    LEFT OUTER JOIN xgb_input r\n" + 
                ") t\n" + 
                "group by rowid, label;")
//@formatter:on
public final class XGBoostPredictTripleUDTF extends XGBoostOnlinePredictUDTF {

    public XGBoostPredictTripleUDTF() {
        super(new Object[3]);
    }

    /** Return (string rowid, int label, double probability) as a result */
    @Override
    protected StructObjectInspector getReturnOI(@Nonnull PrimitiveObjectInspector rowIdOI) {
        List<String> fieldNames = new ArrayList<>(3);
        List<ObjectInspector> fieldOIs = new ArrayList<>(3);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.getPrimitiveWritableObjectInspector(
            rowIdOI.getPrimitiveCategory()));
        fieldNames.add("label");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaIntObjectInspector);
        fieldNames.add("proba");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected void forwardPredicted(@Nonnull Writable rowId, @Nonnull double[] predicted)
            throws HiveException {
        final Object[] forwardObj = _forwardObj;
        forwardObj[0] = rowId;
        for (int j = 0, ncols = predicted.length; j < ncols; j++) {
            forwardObj[1] = Integer.valueOf(j);
            forwardObj[2] = Double.valueOf(predicted[j]);
            forward(forwardObj);
        }
    }

}
