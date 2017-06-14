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
package hivemall;

import hivemall.annotations.VisibleForTesting;
import hivemall.model.FeatureValue;
import hivemall.model.IWeightValue;
import hivemall.model.PredictionModel;
import hivemall.model.WeightValue;
import hivemall.model.WeightValue.WeightValueWithCovar;
import hivemall.optimizer.LossFunctions;
import hivemall.optimizer.LossFunctions.LossFunction;
import hivemall.optimizer.LossFunctions.LossType;
import hivemall.optimizer.Optimizer;
import hivemall.optimizer.OptimizerOptions;
import hivemall.utils.collections.IMapIterator;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.FloatAccumulator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.FloatWritable;

public abstract class GeneralLearnerBaseUDTF extends LearnerBaseUDTF {
    private static final Log logger = LogFactory.getLog(GeneralLearnerBaseUDTF.class);

    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureInputOI;
    private PrimitiveObjectInspector targetOI;
    private boolean parseFeature;

    @Nonnull
    private final Map<String, String> optimizerOptions;
    private Optimizer optimizer;
    private LossFunction lossFunction;

    private PredictionModel model;
    private long count;

    // The accumulated delta of each weight values.
    @Nullable
    private transient Map<Object, FloatAccumulator> accumulated;
    private int sampled;

    private double cumLoss;

    public GeneralLearnerBaseUDTF() {
        this(true);
    }

    public GeneralLearnerBaseUDTF(boolean enableNewModel) {
        super(enableNewModel);
        this.optimizerOptions = OptimizerOptions.create();
    }

    @Nonnull
    protected abstract String getLossOptionDescription();

    @Nonnull
    protected abstract LossType getDefaultLossType();

    protected abstract void checkLossFunction(@Nonnull LossFunction lossFunction)
            throws UDFArgumentException;

    protected abstract void checkTargetValue(float target) throws UDFArgumentException;

    protected abstract void train(@Nonnull final FeatureValue[] features, final float target);

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 2) {
            throw new UDFArgumentException(
                "_FUNC_ takes 2 arguments: List<Int|BigInt|Text> features, float target [, constant string options]");
        }
        this.featureInputOI = processFeaturesOI(argOIs[0]);
        this.targetOI = HiveUtils.asDoubleCompatibleOI(argOIs[1]);

        processOptions(argOIs);

        PrimitiveObjectInspector featureOutputOI = dense_model ? PrimitiveObjectInspectorFactory.javaIntObjectInspector
                : featureInputOI;
        this.model = createModel();
        if (preloadedModelFile != null) {
            loadPredictionModel(model, preloadedModelFile, featureOutputOI);
        }

        try {
            this.optimizer = createOptimizer(optimizerOptions);
        } catch (Throwable e) {
            throw new UDFArgumentException(e);
        }

        this.count = 0L;
        this.sampled = 0;
        this.cumLoss = 0.d;

        return getReturnOI(featureOutputOI);
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("loss", "loss_function", true, getLossOptionDescription());
        OptimizerOptions.setup(opts);
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        LossFunction lossFunction = LossFunctions.getLossFunction(getDefaultLossType());
        if (cl.hasOption("loss_function")) {
            try {
                lossFunction = LossFunctions.getLossFunction(cl.getOptionValue("loss_function"));
            } catch (Throwable e) {
                throw new UDFArgumentException(e.getMessage());
            }
        }
        checkLossFunction(lossFunction);
        this.lossFunction = lossFunction;

        OptimizerOptions.propcessOptions(cl, optimizerOptions);

        return cl;
    }

    @Nonnull
    protected PrimitiveObjectInspector processFeaturesOI(@Nonnull ObjectInspector arg)
            throws UDFArgumentException {
        this.featureListOI = (ListObjectInspector) arg;
        ObjectInspector featureRawOI = featureListOI.getListElementObjectInspector();
        HiveUtils.validateFeatureOI(featureRawOI);
        this.parseFeature = HiveUtils.isStringOI(featureRawOI);
        return HiveUtils.asPrimitiveObjectInspector(featureRawOI);
    }

    @Nonnull
    protected StructObjectInspector getReturnOI(@Nonnull ObjectInspector featureOutputOI) {
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

        fieldNames.add("feature");
        ObjectInspector featureOI = ObjectInspectorUtils.getStandardObjectInspector(featureOutputOI);
        fieldOIs.add(featureOI);
        fieldNames.add("weight");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
        if (useCovariance()) {
            fieldNames.add("covar");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
        }

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (is_mini_batch && accumulated == null) {
            this.accumulated = new HashMap<Object, FloatAccumulator>(1024);
        }

        List<?> features = (List<?>) featureListOI.getList(args[0]);
        FeatureValue[] featureVector = parseFeatures(features);
        if (featureVector == null) {
            return;
        }
        float target = PrimitiveObjectInspectorUtils.getFloat(args[1], targetOI);
        checkTargetValue(target);

        count++;

        train(featureVector, target);
    }

    @Nullable
    public final FeatureValue[] parseFeatures(@Nonnull final List<?> features) {
        final int size = features.size();
        if (size == 0) {
            return null;
        }

        final ObjectInspector featureInspector = featureListOI.getListElementObjectInspector();
        final FeatureValue[] featureVector = new FeatureValue[size];
        for (int i = 0; i < size; i++) {
            Object f = features.get(i);
            if (f == null) {
                continue;
            }
            final FeatureValue fv;
            if (parseFeature) {
                fv = FeatureValue.parse(f);
            } else {
                Object k = ObjectInspectorUtils.copyToStandardObject(f, featureInspector);
                fv = new FeatureValue(k, 1.f);
            }
            featureVector[i] = fv;
        }
        return featureVector;
    }

    public float predict(@Nonnull final FeatureValue[] features) {
        float score = 0.f;
        for (FeatureValue f : features) {// a += w[i] * x[i]
            if (f == null) {
                continue;
            }
            final Object k = f.getFeature();
            final float v = f.getValueAsFloat();

            float old_w = model.getWeight(k);
            if (old_w != 0.f) {
                score += (old_w * v);
            }
        }
        return score;
    }

    protected void update(@Nonnull final FeatureValue[] features, final float target,
            final float predicted) {
        this.cumLoss += lossFunction.loss(predicted, target); // retain cumulative loss to check convergence
        float dloss = lossFunction.dloss(predicted, target);
        if (is_mini_batch) {
            accumulateUpdate(features, dloss);

            if (sampled >= mini_batch_size) {
                batchUpdate();
            }
        } else {
            onlineUpdate(features, dloss);
        }
        optimizer.proceedStep();
    }

    protected void accumulateUpdate(@Nonnull final FeatureValue[] features, final float dloss) {
        for (FeatureValue f : features) {
            Object feature = f.getFeature();
            float xi = f.getValueAsFloat();
            float weight = model.getWeight(feature);

            // compute new weight, but still not set to the model
            float new_weight = optimizer.update(feature, weight, dloss * xi);

            // (w_i - eta * delta_1) + (w_i - eta * delta_2) + ... + (w_i - eta * delta_M)
            FloatAccumulator acc = accumulated.get(feature);
            if (acc == null) {
                acc = new FloatAccumulator(new_weight);
                accumulated.put(feature, acc);
            } else {
                acc.add(new_weight);
            }
        }
        sampled++;
    }

    protected void batchUpdate() {
        if (accumulated.isEmpty()) {
            this.sampled = 0;
            return;
        }

        for (Map.Entry<Object, FloatAccumulator> e : accumulated.entrySet()) {
            Object feature = e.getKey();
            FloatAccumulator v = e.getValue();
            float new_weight = v.get(); // w_i - (eta / M) * (delta_1 + delta_2 + ... + delta_M)
            model.setWeight(feature, new_weight);
        }

        accumulated.clear();
        this.sampled = 0;
    }

    protected void onlineUpdate(@Nonnull final FeatureValue[] features, final float dloss) {
        for (FeatureValue f : features) {
            Object feature = f.getFeature();
            float xi = f.getValueAsFloat();
            float weight = model.getWeight(feature);
            float new_weight = optimizer.update(feature, weight, dloss * xi);
            model.setWeight(feature, new_weight);
        }
    }

    @Override
    public final void close() throws HiveException {
        super.close();
        if (model != null) {
            if (accumulated != null) { // Update model with accumulated delta
                batchUpdate();
                this.accumulated = null;
            }
            int numForwarded = 0;
            if (useCovariance()) {
                final WeightValueWithCovar probe = new WeightValueWithCovar();
                final Object[] forwardMapObj = new Object[3];
                final FloatWritable fv = new FloatWritable();
                final FloatWritable cov = new FloatWritable();
                final IMapIterator<Object, IWeightValue> itor = model.entries();
                while (itor.next() != -1) {
                    itor.getValue(probe);
                    if (!probe.isTouched()) {
                        continue; // skip outputting untouched weights
                    }
                    Object k = itor.getKey();
                    fv.set(probe.get());
                    cov.set(probe.getCovariance());
                    forwardMapObj[0] = k;
                    forwardMapObj[1] = fv;
                    forwardMapObj[2] = cov;
                    forward(forwardMapObj);
                    numForwarded++;
                }
            } else {
                final WeightValue probe = new WeightValue();
                final Object[] forwardMapObj = new Object[2];
                final FloatWritable fv = new FloatWritable();
                final IMapIterator<Object, IWeightValue> itor = model.entries();
                while (itor.next() != -1) {
                    itor.getValue(probe);
                    if (!probe.isTouched()) {
                        continue; // skip outputting untouched weights
                    }
                    Object k = itor.getKey();
                    fv.set(probe.get());
                    forwardMapObj[0] = k;
                    forwardMapObj[1] = fv;
                    forward(forwardMapObj);
                    numForwarded++;
                }
            }
            long numMixed = model.getNumMixed();
            this.model = null;
            logger.info("Trained a prediction model using " + count + " training examples"
                    + (numMixed > 0 ? "( numMixed: " + numMixed + " )" : ""));
            logger.info("Forwarded the prediction model of " + numForwarded + " rows");
        }
    }

    @VisibleForTesting
    public double getCumulativeLoss() {
        return cumLoss;
    }

    @VisibleForTesting
    public void resetCumulativeLoss() {
        this.cumLoss = 0.d;
    }

}
