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
package hivemall.fm;

import hivemall.fm.FFMStringFeatureMapModel.EntryIterator;
import hivemall.fm.FMHyperParameters.FFMHyperParameters;
import hivemall.utils.collections.arrays.DoubleArray3D;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.hadoop.HadoopUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.math.MathUtils;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

/**
 * Field-aware Factorization Machines.
 * 
 * @link https://www.csie.ntu.edu.tw/~cjlin/libffm/
 * @since v0.5-rc.1
 */
@Description(
        name = "train_ffm",
        value = "_FUNC_(array<string> x, double y [, const string options]) - Returns a prediction model")
public final class FieldAwareFactorizationMachineUDTF extends FactorizationMachineUDTF {
    private static final Log LOG = LogFactory.getLog(FieldAwareFactorizationMachineUDTF.class);

    // ----------------------------------------
    // Learning hyper-parameters/options
    private boolean _globalBias;
    private boolean _linearCoeff;

    private int _numFeatures;
    private int _numFields;
    // ----------------------------------------

    private transient FFMStringFeatureMapModel _ffmModel;

    private transient IntArrayList _fieldList;
    @Nullable
    private transient DoubleArray3D _sumVfX;

    public FieldAwareFactorizationMachineUDTF() {
        super();
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("w0", "global_bias", false,
            "Whether to include global bias term w0 [default: OFF]");
        opts.addOption("disable_wi", "no_coeff", false, "Not to include linear term [default: OFF]");
        // feature hashing
        opts.addOption("feature_hashing", true,
            "The number of bits for feature hashing in range [18,31] [default: -1]. No feature hashing for -1.");
        opts.addOption("num_fields", true, "The number of fields [default: 256]");
        // optimizer
        opts.addOption("opt", "optimizer", true,
            "Gradient Descent optimizer [default: ftrl, adagrad, sgd]");
        // adagrad
        opts.addOption("eps", true, "A constant used in the denominator of AdaGrad [default: 1.0]");
        // FTRL
        opts.addOption("alpha", "alphaFTRL", true,
            "Alpha value (learning rate) of Follow-The-Regularized-Reader [default: 0.2]");
        opts.addOption("beta", "betaFTRL", true,
            "Beta value (a learning smoothing parameter) of Follow-The-Regularized-Reader [default: 1.0]");
        opts.addOption(
            "l1",
            "lambda1",
            true,
            "L1 regularization value of Follow-The-Regularized-Reader that controls model Sparseness [default: 0.001]");
        opts.addOption("l2", "lambda2", true,
            "L2 regularization value of Follow-The-Regularized-Reader [default: 0.0001]");
        return opts;
    }

    @Override
    protected boolean isAdaptiveRegularizationSupported() {
        return false;
    }

    @Override
    protected FFMHyperParameters newHyperParameters() {
        return new FFMHyperParameters();
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        FFMHyperParameters params = (FFMHyperParameters) _params;
        this._globalBias = params.globalBias;
        this._linearCoeff = params.linearCoeff;
        this._numFeatures = params.numFeatures;
        this._numFields = params.numFields;

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        StructObjectInspector oi = super.initialize(argOIs);

        this._fieldList = new IntArrayList();
        return oi;
    }

    @Override
    protected StructObjectInspector getOutputOI(@Nonnull FMHyperParameters params) {
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        fieldNames.add("i");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

        fieldNames.add("Wi");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        fieldNames.add("Vi");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableFloatObjectInspector));

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected FFMStringFeatureMapModel initModel(@Nonnull FMHyperParameters params)
            throws UDFArgumentException {
        FFMHyperParameters ffmParams = (FFMHyperParameters) params;

        FFMStringFeatureMapModel model = new FFMStringFeatureMapModel(ffmParams);
        this._ffmModel = model;
        return model;
    }

    @Override
    protected Feature[] parseFeatures(@Nonnull final Object arg) throws HiveException {
        return Feature.parseFFMFeatures(arg, _xOI, _probes, _numFeatures, _numFields);
    }

    @Override
    public void train(@Nonnull final Feature[] x, final double y,
            final boolean adaptiveRegularization) throws HiveException {
        _ffmModel.check(x);
        try {
            trainTheta(x, y);
        } catch (Exception ex) {
            throw new HiveException("Exception caused in the " + _t + "-th call of train()", ex);
        }
    }

    @Override
    protected void trainTheta(@Nonnull final Feature[] x, final double y) throws HiveException {
        final double p = _ffmModel.predict(x);
        final double lossGrad = _ffmModel.dloss(p, y);

        double loss = _lossFunction.loss(p, y);
        _cvState.incrLoss(loss);

        if (MathUtils.closeToZero(lossGrad, 1E-9d)) {
            return;
        }

        // w0 update
        if (_globalBias) {
            float eta_t = _etaEstimator.eta(_t);
            _ffmModel.updateW0(lossGrad, eta_t);
        }

        // ViFf update
        final IntArrayList fieldList = getFieldList(x);
        // sumVfX[i as in index for x][index for field list][index for factorized dimension]
        final DoubleArray3D sumVfX = _ffmModel.sumVfX(x, fieldList, _sumVfX);
        for (int i = 0; i < x.length; i++) {
            final Feature x_i = x[i];
            if (x_i.value == 0.f) {
                continue;
            }
            if (_linearCoeff) {
                _ffmModel.updateWi(lossGrad, x_i, _t);// wi update
            }
            for (int fieldIndex = 0, size = fieldList.size(); fieldIndex < size; fieldIndex++) {
                final int yField = fieldList.get(fieldIndex);
                for (int f = 0, k = _factors; f < k; f++) {
                    final double sumViX = sumVfX.get(i, fieldIndex, f);
                    if (MathUtils.closeToZero(sumViX)) {// grad will be 0 => skip it
                        continue;
                    }
                    _ffmModel.updateV(lossGrad, x_i, yField, f, sumViX, _t);
                }
            }
        }

        // clean up per training instance caches
        sumVfX.clear();
        this._sumVfX = sumVfX;
        fieldList.clear();
    }

    @Nonnull
    private IntArrayList getFieldList(@Nonnull final Feature[] x) {
        for (Feature e : x) {
            int field = e.getField();
            _fieldList.add(field);
        }
        return _fieldList;
    }

    @Override
    protected IntFeature instantiateFeature(@Nonnull final ByteBuffer input) {
        return new IntFeature(input);
    }

    @Override
    public void close() throws HiveException {
        if (LOG.isInfoEnabled()) {
            LOG.info(_ffmModel.getStatistics());
        }

        _ffmModel.disableInitV(); // trick to avoid re-instantiating removed (zero-filled) entry of V
        super.close();

        if (LOG.isInfoEnabled()) {
            LOG.info(_ffmModel.getStatistics());
        }
        this._ffmModel = null;
    }

    @Override
    protected void forwardModel() throws HiveException {
        this._model = null;
        this._fieldList = null;
        this._sumVfX = null;

        final int factors = _factors;
        final IntWritable idx = new IntWritable();
        final FloatWritable Wi = new FloatWritable(0.f);
        final FloatWritable[] Vi = HiveUtils.newFloatArray(factors, 0.f);
        final List<FloatWritable> ViObj = Arrays.asList(Vi);

        final Object[] forwardObjs = new Object[4];
        String modelId = HadoopUtils.getUniqueTaskIdString();
        forwardObjs[0] = new Text(modelId);
        forwardObjs[1] = idx;
        forwardObjs[2] = Wi;
        forwardObjs[3] = null; // Vi

        // W0
        idx.set(0);
        Wi.set(_ffmModel.getW0());
        forward(forwardObjs);

        final EntryIterator itor = _ffmModel.entries();
        final Entry entryW = itor.getEntryProbeW();
        final Entry entryV = itor.getEntryProbeV();
        final float[] Vf = new float[factors];
        while (itor.next()) {
            // set i
            int i = itor.getEntryIndex();
            idx.set(i);

            if (Entry.isEntryW(i)) {// set Wi
                itor.getEntry(entryW);
                float w = entryV.getW();
                if (w == 0.f) {
                    continue; // skip w_i=0
                }
                Wi.set(w);
                forwardObjs[2] = Wi;
                forwardObjs[3] = null;
            } else {// set Vif
                itor.getEntry(entryV);
                entryV.getV(Vf);
                for (int f = 0; f < factors; f++) {
                    Vi[f].set(Vf[f]);
                }
                forwardObjs[2] = null;
                forwardObjs[3] = ViObj;
            }

            forward(forwardObjs);
        }
    }

}
