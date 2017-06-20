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
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NioStatefullSegment;
import hivemall.utils.lang.FloatAccumulator;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnegative;
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
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

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

    // for iterations
    protected NioStatefullSegment fileIO;
    protected ByteBuffer inputBuf;
    private int iterations;
    private double cumLoss;
    private double tol;

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
        opts.addOption("iter", "iterations", true, "The maximum number of iterations [default: 10]");
        opts.addOption("tol", "tolerance", true,
            "Check convergence based on the difference of cumulative loss [default: 1E-1]");
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

        this.iterations = Primitives.parseInt(cl.getOptionValue("iterations"), 10);
        if (iterations < 1) {
            throw new UDFArgumentException(
                "'-iterations' must be greater than or equals to 1: " + iterations);
        }

        this.tol = Primitives.parseDouble(cl.getOptionValue("tolerance"), 1E-1d);

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

        recordTrainSampleToTempFile(featureVector, target);
    }

    protected void recordTrainSampleToTempFile(@Nonnull final FeatureValue[] featureVector, final float target)
            throws HiveException {
        if (iterations == 1) {
            return;
        }

        ByteBuffer buf = inputBuf;
        NioStatefullSegment dst = fileIO;

        if (buf == null) {
            final File file;
            try {
                file = File.createTempFile("hivemall_general_learner", ".sgmt");
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException("Cannot write a temporary file: "
                            + file.getAbsolutePath());
                }
                logger.info("Record training samples to a file: " + file.getAbsolutePath());
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            } catch (Throwable e) {
                throw new UDFArgumentException(e);
            }
            this.inputBuf = buf = ByteBuffer.allocateDirect(1024 * 1024 * 10); // 10 MB
            this.fileIO = dst = new NioStatefullSegment(file, false);
        }

        // feature length, feature 1, feature 2, ..., feature n, target
        int featureVectorBytes = FeatureValue.requiredBytes(featureVector);
        int recordBytes = SizeOf.INT + featureVectorBytes + SizeOf.FLOAT;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(featureVector.length);
        for (FeatureValue f : featureVector) {
            f.writeTo(buf);
        }
        buf.putFloat(target);
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

    private static void writeBuffer(@Nonnull ByteBuffer srcBuf, @Nonnull NioStatefullSegment dst)
            throws HiveException {
        srcBuf.flip();
        try {
            dst.write(srcBuf);
        } catch (IOException e) {
            throw new HiveException("Exception causes while writing a buffer to file", e);
        }
        srcBuf.clear();
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
            if (is_mini_batch) { // Update model with accumulated delta
                batchUpdate();
            }
            if (iterations > 1) {
                runIterativeTraining(iterations);
            }
            forwardModel();
            this.accumulated = null;
            this.model = null;
        }
    }

    protected final void runIterativeTraining(@Nonnegative final int iterations)
            throws HiveException {
        final ByteBuffer buf = this.inputBuf;
        final NioStatefullSegment dst = this.fileIO;
        assert (buf != null);
        assert (dst != null);
        final long numTrainingExamples = count;

        final Reporter reporter = getReporter();
        final Counters.Counter iterCounter = (reporter == null) ? null : reporter.getCounter(
                "hivemall.GeneralLearnerBase$Counter", "iteration");

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();

                int iter = 2;
                double cumLossPrev;
                for (; iter <= iterations; iter++) {
                    cumLossPrev = cumLoss;
                    this.cumLoss = 0.d;

                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        int featureVectorLength = buf.getInt();
                        final FeatureValue[] featureVector = new FeatureValue[featureVectorLength];
                        for (int j = 0; j < featureVectorLength; j++) {
                            featureVector[j] = new FeatureValue(buf);
                        }
                        float target = buf.getFloat();
                        train(featureVector, target);
                    }
                    buf.rewind();

                    if (is_mini_batch) { // Update model with accumulated delta
                        batchUpdate();
                    }

                    logger.info("[iter " + iter + "] cumulative loss: " + cumLoss);

                    if (Math.abs(cumLossPrev - cumLoss) < tol) {
                        break;
                    }
                }
                logger.info("Performed "
                        + Math.min(iter, iterations)
                        + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on memory (thus "
                        + NumberUtils.formatNumber(numTrainingExamples * Math.min(iter, iterations))
                        + " training updates in total) ");
            } else {// read training examples in the temporary file and invoke train for each example
                // write training examples in buffer to a temporary file
                if (buf.remaining() > 0) {
                    writeBuffer(buf, dst);
                }
                try {
                    dst.flush();
                } catch (IOException e) {
                    throw new HiveException("Failed to flush a file: "
                            + dst.getFile().getAbsolutePath(), e);
                }
                if (logger.isInfoEnabled()) {
                    File tmpFile = dst.getFile();
                    logger.info("Wrote " + numTrainingExamples
                            + " records to a temporary file for iterative training: "
                            + tmpFile.getAbsolutePath() + " (" + FileUtils.prettyFileSize(tmpFile)
                            + ")");
                }

                // run iterations
                int iter = 2;
                double cumLossPrev;
                for (; iter <= iterations; iter++) {
                    cumLossPrev = cumLoss;
                    cumLoss = 0.d;

                    setCounterValue(iterCounter, iter);

                    buf.clear();
                    dst.resetPosition();
                    while (true) {
                        reportProgress(reporter);
                        // TODO prefetch
                        // writes training examples to a buffer in the temporary file
                        final int bytesRead;
                        try {
                            bytesRead = dst.read(buf);
                        } catch (IOException e) {
                            throw new HiveException("Failed to read a file: "
                                    + dst.getFile().getAbsolutePath(), e);
                        }
                        if (bytesRead == 0) { // reached file EOF
                            break;
                        }
                        assert (bytesRead > 0) : bytesRead;

                        // reads training examples from a buffer
                        buf.flip();
                        int remain = buf.remaining();
                        if (remain < SizeOf.INT) {
                            throw new HiveException("Illegal file format was detected");
                        }
                        while (remain >= SizeOf.INT) {
                            int pos = buf.position();
                            int recordBytes = buf.getInt();
                            remain -= SizeOf.INT;

                            if (remain < recordBytes) {
                                buf.position(pos);
                                break;
                            }

                            int featureVectorLength = buf.getInt();
                            final FeatureValue[] featureVector = new FeatureValue[featureVectorLength];
                            for (int j = 0; j < featureVectorLength; j++) {
                                featureVector[j] = new FeatureValue(buf);
                            }
                            float target = buf.getFloat();
                            train(featureVector, target);

                            remain -= recordBytes;
                        }
                        buf.compact();
                    }

                    if (is_mini_batch) { // Update model with accumulated delta
                        batchUpdate();
                    }

                    logger.info("[iter " + iter + "] cumulative loss: " + cumLoss);

                    if (Math.abs(cumLossPrev - cumLoss) < tol) {
                        break;
                    }
                }
                logger.info("Performed "
                        + Math.min(iter, iterations)
                        + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on a secondary storage (thus "
                        + NumberUtils.formatNumber(numTrainingExamples * Math.min(iter, iterations))
                        + " training updates in total)");
            }
        } catch (Throwable e) {
            throw new HiveException("Exception caused in the iterative training", e);
        } finally {
            // delete the temporary file and release resources
            try {
                dst.close(true);
            } catch (IOException e) {
                throw new HiveException("Failed to close a file: "
                        + dst.getFile().getAbsolutePath(), e);
            }
            this.inputBuf = null;
            this.fileIO = null;
        }
    }

    protected void forwardModel() throws HiveException {
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
        logger.info("Trained a prediction model using " + count + " training examples"
                + (numMixed > 0 ? "( numMixed: " + numMixed + " )" : ""));
        logger.info("Forwarded the prediction model of " + numForwarded + " rows");
    }

    @VisibleForTesting
    public void closeWithoutModelReset() throws HiveException {
        // launch close(), but not forward & clear model
        super.close();
        if (model != null) {
            if (is_mini_batch) { // Update model with accumulated delta
                batchUpdate();
            }
            if (iterations > 1) {
                runIterativeTraining(iterations);
            }
        }
    }

    @VisibleForTesting
    public double getCumulativeLoss() {
        return cumLoss;
    }
}
