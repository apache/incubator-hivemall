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

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import hivemall.UDTFWithOptions;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.xgboost.utils.XGBoostUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

//@formatter:off
@Description(name = "xgboost_predict",
        value = "_FUNC_(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, array<double> predicted)",
        extended = "select\n" + 
                "  rowid, \n" + 
                "  array_avg(predicted) as predicted,\n" + 
                "  avg(predicted[0]) as predicted0\n" + 
                "from (\n" + 
                "  select\n" + 
                "    xgboost_predict(rowid, features, model_id, model) as (rowid, predicted)\n" + 
                "  from\n" + 
                "    xgb_model l\n" + 
                "    LEFT OUTER JOIN xgb_input r\n" + 
                ") t\n" + 
                "group by rowid;")
//@formatter:on
public class XGBoostOnlinePredictUDTF extends UDTFWithOptions {

    // For input parameters
    private PrimitiveObjectInspector rowIdOI;
    private ListObjectInspector featureListOI;
    private boolean denseFeatures;
    @Nullable
    private PrimitiveObjectInspector featureElemOI;
    private StringObjectInspector modelIdOI;
    private StringObjectInspector modelOI;

    // For input buffer
    @Nullable
    private transient Map<String, Predictor> mapToModel;

    @Nonnull
    protected transient final Object[] _forwardObj;
    @Nullable
    protected transient List<DoubleWritable> _predictedCache;

    public XGBoostOnlinePredictUDTF() {
        this(new Object[2]);
    }

    protected XGBoostOnlinePredictUDTF(@Nonnull Object[] forwardObj) {
        super();
        this._forwardObj = forwardObj;
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        // not yet supported
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs, 4);
            cl = parseOptions(rawArgs);
        }
        return cl;
    }

    @Override
    public StructObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 4 && argOIs.length != 5) {
            showHelp("Invalid argment length=" + argOIs.length);
        }
        processOptions(argOIs);

        this.rowIdOI = HiveUtils.asPrimitiveObjectInspector(argOIs, 0);
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs, 1);
        this.featureListOI = listOI;
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.denseFeatures = true;
        } else if (HiveUtils.isStringOI(elemOI)) {
            this.denseFeatures = false;
        } else {
            throw new UDFArgumentException(
                "Expected array<string|double> for the 2nd argment but got an unexpected features type: "
                        + listOI.getTypeName());
        }
        this.modelIdOI = HiveUtils.asStringOI(argOIs, 2);
        this.modelOI = HiveUtils.asStringOI(argOIs, 3);
        return getReturnOI(rowIdOI);
    }

    /** Override this to output predicted results depending on a task type */
    /** Return (primitive rowid, array<double> predicted) as a result */
    @Nonnull
    protected StructObjectInspector getReturnOI(@Nonnull PrimitiveObjectInspector rowIdOI) {
        List<String> fieldNames = new ArrayList<>(2);
        List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.getPrimitiveWritableObjectInspector(
            rowIdOI.getPrimitiveCategory()));
        fieldNames.add("predicted");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (mapToModel == null) {
            this.mapToModel = new HashMap<String, Predictor>();
        }
        if (args[1] == null) {// features is null
            return;
        }

        String modelId =
                PrimitiveObjectInspectorUtils.getString(nonNullArgument(args, 2), modelIdOI);
        Predictor model = mapToModel.get(modelId);
        if (model == null) {
            Text arg3 = modelOI.getPrimitiveWritableObject(nonNullArgument(args, 3));
            model = XGBoostUtils.loadPredictor(arg3);
            mapToModel.put(modelId, model);
        }

        Writable rowId = HiveUtils.copyToWritable(nonNullArgument(args, 0), rowIdOI);
        FVec features = denseFeatures ? parseDenseFeatures(args[1])
                : parseSparseFeatures(featureListOI.getList(args[1]));

        predictAndForward(model, rowId, features);
    }

    @Nonnull
    private FVec parseDenseFeatures(@Nonnull Object argObj) throws UDFArgumentException {
        final int length = featureListOI.getListLength(argObj);
        final double[] values = new double[length];
        for (int i = 0; i < length; i++) {
            final Object o = featureListOI.getListElement(argObj, i);
            final double v;
            if (o == null) {
                v = Double.NaN;
            } else {
                v = PrimitiveObjectInspectorUtils.getDouble(o, featureElemOI);
            }
            values[i] = v;

        }
        return FVec.Transformer.fromArray(values, false);
    }

    @Nonnull
    private static FVec parseSparseFeatures(@Nonnull final List<?> featureList)
            throws UDFArgumentException {
        final Map<Integer, Double> map = new HashMap<>((int) (featureList.size() * 1.5));
        for (Object f : featureList) {
            if (f == null) {
                continue;
            }
            String str = f.toString();
            final int pos = str.indexOf(':');
            if (pos < 1) {
                throw new UDFArgumentException("Invalid feature format: " + str);
            }
            final int index;
            final double value;
            try {
                index = Integer.parseInt(str.substring(0, pos));
                value = Double.parseDouble(str.substring(pos + 1));
            } catch (NumberFormatException e) {
                throw new UDFArgumentException("Failed to parse a feature value: " + str);
            }
            map.put(index, value);
        }

        return FVec.Transformer.fromMap(map);
    }

    private void predictAndForward(@Nonnull final Predictor model, @Nonnull final Writable rowId,
            @Nonnull final FVec features) throws HiveException {
        double[] predicted = model.predict(features);
        // predicted[0] has
        //    - probability ("binary:logistic")
        //    - class label ("multi:softmax")
        forwardPredicted(rowId, predicted);
    }

    protected void forwardPredicted(@Nonnull final Writable rowId,
            @Nonnull final double[] predicted) throws HiveException {
        List<DoubleWritable> list = WritableUtils.toWritableList(predicted, _predictedCache);
        this._predictedCache = list;
        Object[] forwardObj = this._forwardObj;
        forwardObj[0] = rowId;
        forwardObj[1] = list;
        forward(forwardObj);
    }

    @Override
    public void close() throws HiveException {
        this.mapToModel = null;
    }

}
