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
import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.lang.Primitives;
import hivemall.utils.struct.Pair;
import hivemall.xgboost.utils.NativeLibLoader;

import java.io.IOException;
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
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.Text;

public abstract class XGBoostPredictUDTF extends UDTFWithOptions {

    // For input parameters
    private StringObjectInspector rowIdOI;
    private ListObjectInspector featureListOI;
    private StringObjectInspector modelIdOI;
    private StringObjectInspector modelOI;

    private int batchSize;

    // For input buffer
    @Nullable
    private transient Map<String, Predictor> mapToModel;
    @Nullable
    private transient Map<String, List<Pair<String, FVec>>> rowBuffer;

    // Settings for the XGBoost native library
    static {
        NativeLibLoader.initXGBoost();
    }

    public XGBoostPredictUDTF() {
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
                throw new UDFArgumentException("batch_size must be greater than 0: " + batchSize);
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
                    + " takes 4 or 5 arguments: string rowid, array<string> features, string model_id,"
                    + " array<byte> pred_model [, string options]: " + argOIs.length);
        }
        this.processOptions(argOIs);
        this.rowIdOI = HiveUtils.asStringOI(argOIs[0]);
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[1]);
        this.featureListOI = listOI;
        this.modelIdOI = HiveUtils.asStringOI(argOIs[2]);
        this.modelOI = HiveUtils.asStringOI(argOIs[3]);
        return getReturnOI();
    }

    /** Override this to output predicted results depending on a task type */
    @Nonnull
    protected abstract StructObjectInspector getReturnOI();

    @Override
    public void process(Object[] args) throws HiveException {
        if (mapToModel == null) {
            this.mapToModel = new HashMap<String, Predictor>();
            this.rowBuffer = new HashMap<String, List<Pair<String, FVec>>>();
        }
        if (args[1] == null) {// features is null
            return;
        }

        String modelId =
                PrimitiveObjectInspectorUtils.getString(nonNullArgument(args, 2), modelIdOI);
        Predictor model = mapToModel.get(modelId);
        if (model == null) {
            Text arg3 = modelOI.getPrimitiveWritableObject(nonNullArgument(args, 3));
            try {
                model = new Predictor(
                    new FastByteArrayInputStream(arg3.getBytes(), 0, arg3.getLength()));
            } catch (IOException e) {
                throw new HiveException("Failed to load a model: " + modelId, e);
            }
            mapToModel.put(modelId, model);
        }

        List<Pair<String, FVec>> rowBatch = rowBuffer.get(modelId);
        if (rowBatch == null) {
            rowBatch = new ArrayList<>(batchSize);
            rowBuffer.put(modelId, rowBatch);
        }

        String rowId = PrimitiveObjectInspectorUtils.getString(nonNullArgument(args, 0), rowIdOI);
        List<?> featureList = featureListOI.getList(args[1]);
        Pair<String, FVec> row = parseRow(rowId, featureList);
        rowBatch.add(row);

        if (rowBuffer.size() >= batchSize) {
            predictAndFlush(model, rowBatch);
        }
    }

    @Nonnull
    protected static Pair<String, FVec> parseRow(@Nonnull final String rowId,
            @Nonnull final List<?> featureList) throws UDFArgumentException {
        final Map<Integer, Float> map = new HashMap<>((int) (featureList.size() * 1.5));
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
            final float value;
            try {
                index = Integer.parseInt(str.substring(0, pos));
                value = Float.parseFloat(str.substring(pos + 1));
            } catch (NumberFormatException e) {
                throw new UDFArgumentException("Failed to parse a feature value: " + str);
            }
            map.put(index, value);
        }

        FVec fvec = FVec.Transformer.fromMap(map);
        return new Pair<>(rowId, fvec);
    }

    @Override
    public void close() throws HiveException {
        for (Entry<String, List<Pair<String, FVec>>> r : rowBuffer.entrySet()) {
            String modelId = r.getKey();
            Predictor model = Objects.requireNonNull(mapToModel.get(modelId));
            List<Pair<String, FVec>> rowBatch = r.getValue();
            predictAndFlush(model, rowBatch);
        }
    }

    private void predictAndFlush(final Predictor model, final List<Pair<String, FVec>> rowBatch)
            throws HiveException {
        for (Pair<String, FVec> r : rowBatch) {
            String rowId = r.getKey();
            FVec features = r.getValue();
            double[] predicted = model.predict(features);
            // prediction[0] has
            //    - probability ("binary:logistic")
            //    - class label ("multi:softmax")
            forwardPredicted(rowId, predicted);
        }
        rowBatch.clear();
    }

    protected abstract void forwardPredicted(@Nonnull String rowId, @Nonnull double[] predicted)
            throws HiveException;

}
