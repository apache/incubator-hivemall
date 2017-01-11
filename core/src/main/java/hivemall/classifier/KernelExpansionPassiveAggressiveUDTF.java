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
package hivemall.classifier;

import hivemall.common.LossFunctions;
import hivemall.model.FeatureValue;
import hivemall.model.PredictionModel;
import hivemall.model.PredictionResult;
import hivemall.utils.collections.Int2FloatOpenHashTable;
import hivemall.utils.collections.Int2FloatOpenHashTable.IMapIterator;
import hivemall.utils.hashing.HashFunction;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;

/**
 * Degree-2 polynomial kernel expansion Passive Aggressive.
 * 
 * <pre>
 * Hideki Isozaki and Hideto Kazawa: Efficient Support Vector Classifiers for Named Entity Recognition, Proc.COLING, 2002
 * </pre>
 */
@Description(name = "train_kpa",
        value = "_FUNC_(array<string|int|bigint> features, int label [, const string options])"
                + " - returns a relation <f string, float w0, float w1, float w2, float w3>")
public final class KernelExpansionPassiveAggressiveUDTF extends BinaryOnlineClassifierUDTF {

    // ------------------------------------
    // Hyper parameters
    private float _pkc;
    //PA1, PA2
    private boolean _pa1, _pa2;
    private float _c;

    // ------------------------------------
    // Model parameters

    private float _w0;
    private Int2FloatOpenHashTable _w1;
    private Int2FloatOpenHashTable _w2;
    private Int2FloatOpenHashTable _w3;

    // ------------------------------------

    private float _loss;

    public KernelExpansionPassiveAggressiveUDTF() {}

    //@VisibleForTesting
    float getLoss() {//only used for testing purposes at the moment
        return _loss;
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("pkc", true,
            "Constant c inside polynomial kernel K = (dot(xi,xj) + c)^2 [default 1.0]");
        opts.addOption("pa1", false, "Whether to use PA-1 for calculating loss");
        opts.addOption("pa2", false, "Whether to use PA-2 for calculating loss");
        opts.addOption("c", "aggressiveness", true,
            "Aggressiveness parameter C for PA-1 and PA-2 [default 1.0]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        this._pa1 = false;
        this._pa2 = false;
        float pkc = 1.f;
        float c = 1.f;

        final CommandLine cl = super.processOptions(argOIs);
        if (cl != null) {
            this._pa1 = cl.hasOption("pa1");
            this._pa2 = cl.hasOption("pa2");
            if (_pa1 && _pa2) {
                throw new UDFArgumentException("PA-1 and PA-2 cannot be simultaneously set.");
            }
            String pkc_str = cl.getOptionValue("pkc");
            if (pkc_str != null) {
                pkc = Float.parseFloat(pkc_str);
            }
            String c_str = cl.getOptionValue("c");
            if (c_str != null) {
                c = Float.parseFloat(c_str);
                if (!(c > 0.f)) {
                    throw new UDFArgumentException("Aggressiveness parameter C must be C > 0: " + c);
                }
            }
        }
        this._pkc = pkc;
        this._c = c;

        return cl;
    }

    @Override
    protected PredictionModel createModel() {
        this._w0 = 0.f;
        this._w1 = new Int2FloatOpenHashTable(16384);
        _w1.defaultReturnValue(0.f);
        this._w2 = new Int2FloatOpenHashTable(16384);
        _w2.defaultReturnValue(0.f);
        this._w3 = new Int2FloatOpenHashTable(16384);
        _w3.defaultReturnValue(0.f);

        return null;
    }

    @Override
    protected StructObjectInspector getReturnOI(ObjectInspector featureRawOI) {
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

        fieldNames.add("f");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("w0");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
        fieldNames.add("w1");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
        fieldNames.add("w2");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
        fieldNames.add("w3");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Nullable
    protected FeatureValue[] parseFeatures(@Nonnull final List<?> features) {
        final int size = features.size();
        if (size == 0) {
            return null;
        }

        final FeatureValue[] featureVector = new FeatureValue[size];
        for (int i = 0; i < size; i++) {
            Object f = features.get(i);
            if (f == null) {
                continue;
            }
            FeatureValue fv = FeatureValue.parse(f, true);
            featureVector[i] = fv;
        }
        return featureVector;
    }

    @Override
    protected void train(@Nonnull final FeatureValue[] features, final int label) {
        final float y = label > 0 ? 1.f : -1.f;

        PredictionResult margin = calcScoreWithKernelAndNorm(features);
        float p = margin.getScore();
        float loss = LossFunctions.hingeLoss(p, y); // 1.0 - y * p
        this._loss = loss;

        if (loss > 0.f) { // y * p < 1
            updateKernel(y, loss, margin, features);
        }
    }

    @Nonnull
    final PredictionResult calcScoreWithKernelAndNorm(@Nonnull final FeatureValue[] features) {
        float score = _w0;
        float norm = 0.f;
        for (int i = 0; i < features.length; ++i) {
            if (features[i] == null) {
                continue;
            }
            int h = features[i].getFeatureAsInt();
            float w1 = _w1.get(h);
            float w2 = _w2.get(h);
            double xi = features[i].getValue();
            double xx = xi * xi;
            score += w1 * xi;
            score += w2 * xx;
            norm += xx;
            for (int j = i + 1; j < features.length; ++j) {
                int k = features[j].getFeatureAsInt();
                int hk = HashFunction.hash(h, k);
                float w3 = _w3.get(hk);
                double xj = features[j].getValue();
                score += xi * xj * w3;
            }
        }
        return new PredictionResult(score).squaredNorm(norm);
    }

    protected void updateKernel(final float label, final float loss,
            @Nonnull final PredictionResult margin, @Nonnull final FeatureValue[] features) {
        float eta = _pa1 ? eta1(loss, margin) : _pa2 ? eta2(loss, margin) : eta(loss, margin);
        float diff = eta * label;
        expandKernel(features, diff);
    }

    /**
     * @return learning rate for PA
     */
    protected float eta(final float loss, @Nonnull final PredictionResult margin) {
        return loss / margin.getSquaredNorm();
    }

    /**
     * @return learning rate for PA1
     */
    protected float eta1(final float loss, @Nonnull final PredictionResult margin) {
        float squared_norm = margin.getSquaredNorm();
        float eta = loss / squared_norm;
        return Math.min(_c, eta);
    }

    /**
     * @return learning rate for PA2
     */
    protected float eta2(final float loss, @Nonnull final PredictionResult margin) {
        float squared_norm = margin.getSquaredNorm();
        float eta = loss / (squared_norm + (0.5f / _c));
        return eta;
    }

    private void expandKernel(@Nonnull final FeatureValue[] supportVector, final float alpha) {
        // W0 += α c^2
        this._w0 += alpha * _pkc * _pkc;

        for (int i = 0; i < supportVector.length; ++i) {
            final FeatureValue si = supportVector[i];
            final int h = si.getFeatureAsInt();
            float Zih = si.getValueAsFloat();

            // W1[h] += 2 c α Zi[h]
            _w1.put(h, _w1.get(h) + 2.f * _pkc * alpha * Zih);
            // W2[h] += α Zi[h]^2
            _w2.put(h, _w2.get(h) + alpha * Zih * Zih);

            for (int j = i + 1; j < supportVector.length; ++j) {
                FeatureValue sj = supportVector[j];
                int k = sj.getFeatureAsInt();
                int hk = HashFunction.hash(h, k);
                float Zjk = sj.getValueAsFloat();

                // W3 += 2 α Zi[h] Zi[k]
                _w3.put(hk, _w3.get(hk) + 2.f * alpha * Zih * Zjk);
            }
        }
    }

    @Override
    public void close() throws HiveException {
        final IntWritable f = new IntWritable();
        final FloatWritable w0 = new FloatWritable(_w0);
        final FloatWritable w1 = new FloatWritable();
        final FloatWritable w2 = new FloatWritable();
        final FloatWritable w3 = new FloatWritable();
        final Object[] row = new Object[] {f, w0, w1, w2, null};

        final Int2FloatOpenHashTable w2map = _w2;
        final IMapIterator w1itor = _w1.entries();
        while (w1itor.next() != -1) {
            int h = w1itor.getKey();
            f.set(h);
            w1.set(w1itor.getValue());
            w2.set(w2map.get(h));
            forward(row);
            row[1] = null; // w0 is forwarded once
        }
        this._w1 = null;
        this._w2 = null;

        row[2] = null;
        row[3] = null;
        row[4] = w3;
        final IMapIterator w3itor = _w3.entries();
        while (w3itor.next() != -1) {
            int hk = w3itor.getKey();
            f.set(hk);
            w3.set(w3itor.getValue());
            forward(row);
            row[1] = null; // w0 is forwarded once
        }
        this._w3 = null;
    }

}
