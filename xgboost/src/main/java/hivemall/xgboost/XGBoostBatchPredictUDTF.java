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

import hivemall.UDTFWithOptions;
import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Primitives;
import hivemall.xgboost.utils.NativeLibLoader;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

//@formatter:off
@Description(name = "xgboost_batch_predict",
        value = "_FUNC_(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options]) "
                + "- Returns a prediction result as (string rowid, array<double> predicted)",
        extended = "select\n" + 
                "  rowid, \n" + 
                "  array_avg(predicted) as predicted,\n" + 
                "  avg(predicted[0]) as predicted0\n" + 
                "from (\n" + 
                "  select\n" + 
                "    xgboost_batch_predict(rowid, features, model_id, model) as (rowid, predicted)\n" + 
                "  from\n" + 
                "    xgb_model l\n" + 
                "    LEFT OUTER JOIN xgb_input r\n" + 
                ") t\n" + 
                "group by rowid;")
//@formatter:on
public final class XGBoostBatchPredictUDTF extends UDTFWithOptions {

    // For input parameters
    private PrimitiveObjectInspector rowIdOI;
    private ListObjectInspector featureListOI;
    private boolean denseFeatures;
    @Nullable
    private PrimitiveObjectInspector featureElemOI;
    private StringObjectInspector modelIdOI;
    private StringObjectInspector modelOI;

    // For input buffer
    private transient Map<String, Booster> mapToModel;
    private transient Map<String, List<LabeledPointWithRowId>> rowBuffer;

    private int _batchSize;

    @Nonnull
    protected transient final Object[] _forwardObj;

    // Settings for the XGBoost native library
    static {
        NativeLibLoader.initXGBoost();
    }

    public XGBoostBatchPredictUDTF() {
        super();
        this._forwardObj = new Object[2];
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("batch_size", true, "Number of rows to predict together [default: 128]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        int batchSize = 128;
        CommandLine cl = null;
        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs, 4);
            cl = parseOptions(rawArgs);
            batchSize = Primitives.parseInt(cl.getOptionValue("batch_size"), batchSize);
            if (batchSize < 1) {
                throw new UDFArgumentException("batch_size must be greater than 0: " + batchSize);
            }
        }
        this._batchSize = batchSize;
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

        this.featureListOI = HiveUtils.asListOI(argOIs, 1);
        ObjectInspector elemOI = featureListOI.getListElementObjectInspector();
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.denseFeatures = true;
        } else if (HiveUtils.isStringOI(elemOI)) {
            this.denseFeatures = false;
        } else {
            throw new UDFArgumentException(
                "Expected array<string|double> for the 2nd argment but got an unexpected features type: "
                        + featureListOI.getTypeName());
        }
        this.modelIdOI = HiveUtils.asStringOI(argOIs, 2);
        this.modelOI = HiveUtils.asStringOI(argOIs, 3);

        return getReturnOI(rowIdOI);
    }

    /** Override this to output predicted results depending on a task type */
    /** Return (string rowid, array<double> predicted) as a result */
    @Nonnull
    protected StructObjectInspector getReturnOI(@Nonnull PrimitiveObjectInspector rowIdOI) {
        List<String> fieldNames = new ArrayList<>(2);
        List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("rowid");
        fieldOIs.add(PrimitiveObjectInspectorFactory.getPrimitiveWritableObjectInspector(
            rowIdOI.getPrimitiveCategory()));
        fieldNames.add("predicted");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableFloatObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (mapToModel == null) {
            this.mapToModel = new HashMap<String, Booster>();
            this.rowBuffer = new HashMap<String, List<LabeledPointWithRowId>>();
        }
        if (args[1] == null) {
            return;
        }

        String modelId =
                PrimitiveObjectInspectorUtils.getString(nonNullArgument(args, 2), modelIdOI);
        Booster model = mapToModel.get(modelId);
        if (model == null) {
            Text arg3 = modelOI.getPrimitiveWritableObject(nonNullArgument(args, 3));
            model = XGBoostUtils.deserializeBooster(arg3);
            mapToModel.put(modelId, model);
        }

        List<LabeledPointWithRowId> rowBatch = rowBuffer.get(modelId);
        if (rowBatch == null) {
            rowBatch = new ArrayList<LabeledPointWithRowId>(_batchSize);
            rowBuffer.put(modelId, rowBatch);
        }
        LabeledPointWithRowId row = parseRow(args);
        rowBatch.add(row);
        if (rowBatch.size() >= _batchSize) {
            predictAndFlush(model, rowBatch);
        }
    }

    @Nonnull
    private LabeledPointWithRowId parseRow(@Nonnull Object[] args) throws UDFArgumentException {
        final Writable rowId = HiveUtils.copyToWritable(nonNullArgument(args, 0), rowIdOI);

        final Object arg1 = args[1];
        if (denseFeatures) {
            return parseDenseFeatures(rowId, arg1, featureListOI, featureElemOI);
        } else {
            return parseSparseFeatures(rowId, arg1, featureListOI);
        }
    }

    @Nonnull
    private static LabeledPointWithRowId parseDenseFeatures(@Nonnull final Writable rowId,
            @Nonnull final Object argObj, @Nonnull final ListObjectInspector featureListOI,
            @Nonnull final PrimitiveObjectInspector featureElemOI) throws UDFArgumentException {
        final int size = featureListOI.getListLength(argObj);

        final float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            final Object o = featureListOI.getListElement(argObj, i);
            if (o == null) {
                values[i] = Float.NaN;
            } else {
                float v = PrimitiveObjectInspectorUtils.getFloat(o, featureElemOI);
                values[i] = v;
            }
        }

        return new LabeledPointWithRowId(rowId, /* dummy label */ 0.f, null, values);

    }

    @Nonnull
    private static LabeledPointWithRowId parseSparseFeatures(@Nonnull final Writable rowId,
            @Nonnull final Object argObj, @Nonnull final ListObjectInspector featureListOI)
            throws UDFArgumentException {
        final int size = featureListOI.getListLength(argObj);
        final IntArrayList indices = new IntArrayList(size);
        final FloatArrayList values = new FloatArrayList(size);

        for (int i = 0; i < size; i++) {
            Object f = featureListOI.getListElement(argObj, i);
            if (f == null) {
                continue;
            }
            final String str = f.toString();
            final int pos = str.indexOf(':');
            if (pos < 1) {
                throw new UDFArgumentException("Invalid feature format: " + str);
            }
            final int index;
            final float value;
            try {
                index = Integer.parseInt(str.substring(0, pos));
                value = Float.parseFloat(str.substring(pos + 1));
            } catch (NumberFormatException e) {
                throw new UDFArgumentException("Failed to parse a feature value: " + str);
            }
            indices.add(index);
            values.add(value);
        }

        return new LabeledPointWithRowId(rowId, /* dummy label */ 0.f, indices.toArray(),
            values.toArray());
    }


    @Override
    public void close() throws HiveException {
        for (Entry<String, List<LabeledPointWithRowId>> e : rowBuffer.entrySet()) {
            String modelId = e.getKey();
            List<LabeledPointWithRowId> rowBatch = e.getValue();
            if (rowBatch.isEmpty()) {
                continue;
            }
            final Booster model = Objects.requireNonNull(mapToModel.get(modelId));
            try {
                predictAndFlush(model, rowBatch);
            } finally {
                XGBoostUtils.close(model);
            }
        }
        this.rowBuffer = null;
        this.mapToModel = null;
    }

    private void predictAndFlush(@Nonnull final Booster model,
            @Nonnull final List<LabeledPointWithRowId> rowBatch) throws HiveException {
        DMatrix testData = null;
        final float[][] predicted;
        try {
            testData = XGBoostUtils.createDMatrix(rowBatch);
            predicted = model.predict(testData);
        } catch (XGBoostError e) {
            throw new HiveException("Exception caused at prediction", e);
        } finally {
            XGBoostUtils.close(testData);
        }
        forwardPredicted(rowBatch, predicted);
        rowBatch.clear();
    }

    private void forwardPredicted(@Nonnull final List<LabeledPointWithRowId> rowBatch,
            @Nonnull final float[][] predicted) throws HiveException {
        if (rowBatch.size() != predicted.length) {
            throw new HiveException(String.format("buf.size() = %d but predicted.length = %d",
                rowBatch.size(), predicted.length));
        }
        if (predicted.length == 0) {
            return;
        }

        final int ncols = predicted[0].length;
        final List<FloatWritable> list = WritableUtils.newFloatList(ncols);

        final Object[] forwardObj = this._forwardObj;
        forwardObj[1] = list;

        for (int i = 0; i < predicted.length; i++) {
            Writable rowId = Objects.requireNonNull(rowBatch.get(i)).getRowId();
            forwardObj[0] = rowId;
            WritableUtils.setValues(predicted[i], list);
            forward(forwardObj);
        }
    }

    public static final class LabeledPointWithRowId extends LabeledPoint {
        private static final long serialVersionUID = -7150841669515184648L;

        @Nonnull
        final Writable rowId;

        LabeledPointWithRowId(@Nonnull Writable rowId, float label, @Nullable int[] indices,
                @Nonnull float[] values) {
            super(label, indices, values);
            this.rowId = rowId;
        }

        @Nonnull
        public Writable getRowId() {
            return rowId;
        }
    }

}
