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
import hivemall.utils.hadoop.HiveUtils;
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

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

public abstract class XGBoostPredictBaseUDTF extends UDTFWithOptions {

    // For input parameters
    private PrimitiveObjectInspector rowIdOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector modelIdOI;
    private PrimitiveObjectInspector modelOI;

    // For input buffer
    private Map<String, Booster> mapToModel;
    private Map<String, List<LabeledPointWithRowId>> rowBuffer;

    private int batchSize;

    // Settings for the XGBoost native library
    static {
        NativeLibLoader.initXGBoost();
    }

    public XGBoostPredictBaseUDTF() {
        super();
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
            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);
            batchSize = Primitives.parseInt(cl.getOptionValue("batch_size"), batchSize);
            if (batchSize < 1) {
                throw new IllegalArgumentException(
                    "batch_size must be greater than 0: " + batchSize);
            }
        }
        this.batchSize = batchSize;
        return cl;
    }

    @Override
    public StructObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 4 && argOIs.length != 5) {
            throw new UDFArgumentException(this.getClass().getSimpleName()
                    + " takes 4 or 5 arguments: string rowid, string[] features, string model_id,"
                    + " array<byte> pred_model [, string options]: " + argOIs.length);
        } else {
            this.processOptions(argOIs);
            this.rowIdOI = HiveUtils.asStringOI(argOIs[0]);
            ListObjectInspector listOI = HiveUtils.asListOI(argOIs[1]);
            this.featureListOI = listOI;
            this.modelIdOI = HiveUtils.asStringOI(argOIs[2]);
            this.modelOI = HiveUtils.asBinaryOI(argOIs[3]);
            this.mapToModel = new HashMap<String, Booster>();
            this.rowBuffer = new HashMap<String, List<LabeledPointWithRowId>>();
            return getReturnOI();
        }
    }

    /** Override this to output predicted results depending on a task type */
    @Nonnull
    protected abstract StructObjectInspector getReturnOI();

    @Override
    public void process(Object[] args) throws HiveException {
        if (args[1] == null) {
            return;
        }

        final String modelId = PrimitiveObjectInspectorUtils.getString(args[2], modelIdOI);
        if (!mapToModel.containsKey(modelId)) {
            byte[] predModel = PrimitiveObjectInspectorUtils.getBinary(args[3], modelOI).getBytes();
            mapToModel.put(modelId, XGBoostUtils.loadBooster(predModel));
        }

        List<LabeledPointWithRowId> buf = rowBuffer.get(modelId);
        if (buf == null) {
            buf = new ArrayList<LabeledPointWithRowId>();
            rowBuffer.put(modelId, buf);
        }

        final LabeledPointWithRowId point = parseFeatures(args);
        buf.add(point);
        if (buf.size() >= batchSize) {
            predictAndFlush(mapToModel.get(modelId), buf);
        }
    }

    @Nonnull
    protected LabeledPointWithRowId parseFeatures(@Nonnull final Object[] args) {
        final String rowId = PrimitiveObjectInspectorUtils.getString(args[0], rowIdOI);

        final List<?> features = featureListOI.getList(args[1]);
        final int size = features.size();
        final int[] indices = new int[size];
        final float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            Object f = features.get(i);
            if (f == null) {
                continue;
            }
            String str = f.toString();
            final int pos = str.indexOf(':');
            if (pos >= 1) {
                indices[i] = Integer.parseInt(str.substring(0, pos));
                values[i] = Float.parseFloat(str.substring(pos + 1));
            }
        }

        return new LabeledPointWithRowId(rowId, 0.f, indices, values);
    }

    @Override
    public void close() throws HiveException {
        for (Entry<String, List<LabeledPointWithRowId>> e : rowBuffer.entrySet()) {
            predictAndFlush(mapToModel.get(e.getKey()), e.getValue());
        }
    }

    private void predictAndFlush(final Booster model, final List<LabeledPointWithRowId> buf)
            throws HiveException {
        final DMatrix testData;
        final float[][] predicted;
        try {
            testData = XGBoostUtils.createDMatrix(buf);
            predicted = model.predict(testData);
        } catch (XGBoostError e) {
            throw new HiveException(e);
        }
        forwardPredicted(buf, predicted);
        buf.clear();
    }

    protected abstract void forwardPredicted(@Nonnull final List<LabeledPointWithRowId> testData,
            @Nonnull final float[][] predicted) throws HiveException;

    public static final class LabeledPointWithRowId extends LabeledPoint {
        private static final long serialVersionUID = 2227408319743631799L;

        @Nonnull
        final String rowId;

        public LabeledPointWithRowId(@Nonnull String rowId, float label, @Nonnull int[] indices,
                @Nonnull float[] values) {
            super(label, indices, values);
            this.rowId = rowId;
        }

        @Nonnull
        public String getRowId() {
            return rowId;
        }
    }

}
