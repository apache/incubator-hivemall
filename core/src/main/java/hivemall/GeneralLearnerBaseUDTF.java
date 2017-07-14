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
import hivemall.common.ConversionState;
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
import hivemall.utils.io.NIOUtils;
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
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

public abstract class GeneralLearnerBaseUDTF extends LearnerBaseUDTF {
    private static final Log logger = LogFactory.getLog(GeneralLearnerBaseUDTF.class);

    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector targetOI;
    private FeatureType featureType;

    // -----------------------------------------
    // hyperparameters

    @Nonnull
    private final Map<String, String> optimizerOptions;
    private Optimizer optimizer;
    private LossFunction lossFunction;

    // -----------------------------------------

    private PredictionModel model;
    private long count;

    // -----------------------------------------
    // for mini-batch

    /** The accumulated delta of each weight values. */
    @Nullable
    private transient Map<Object, FloatAccumulator> accumulated;
    private int sampled;

    // -----------------------------------------
    // for iterations

    @Nullable
    protected transient NioStatefullSegment fileIO;
    @Nullable
    protected transient ByteBuffer inputBuf;
    private int iterations;
    protected ConversionState cvState;

    // -----------------------------------------

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
        this.featureListOI = HiveUtils.asListOI(argOIs[0]);
        this.featureType = getFeatureType(featureListOI);
        this.targetOI = HiveUtils.asDoubleCompatibleOI(argOIs[1]);

        processOptions(argOIs);

        this.model = createModel();

        try {
            this.optimizer = createOptimizer(optimizerOptions);
        } catch (Throwable e) {
            throw new UDFArgumentException(e);
        }

        this.count = 0L;
        this.sampled = 0;

        return getReturnOI(getFeatureOutputOI(featureType));
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("loss", "loss_function", true, getLossOptionDescription());
        opts.addOption("iter", "iterations", true, "The maximum number of iterations [default: 10]");
        // conversion check
        opts.addOption("disable_cv", "disable_cvtest", false,
            "Whether to disable convergence check [default: OFF]");
        opts.addOption("cv_rate", "convergence_rate", true,
            "Threshold to determine convergence [default: 0.005]");
        OptimizerOptions.setup(opts);
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        LossFunction lossFunction = LossFunctions.getLossFunction(getDefaultLossType());
        int iterations = 10;
        boolean conversionCheck = true;
        double convergenceRate = 0.005d;

        if (cl != null) {
            if (cl.hasOption("loss_function")) {
                try {
                    lossFunction = LossFunctions.getLossFunction(cl.getOptionValue("loss_function"));
                } catch (Throwable e) {
                    throw new UDFArgumentException(e.getMessage());
                }
            }
            checkLossFunction(lossFunction);

            iterations = Primitives.parseInt(cl.getOptionValue("iterations"), iterations);
            if (iterations < 1) {
                throw new UDFArgumentException(
                    "'-iterations' must be greater than or equals to 1: " + iterations);
            }

            conversionCheck = !cl.hasOption("disable_cvtest");
            convergenceRate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), convergenceRate);
        }

        this.lossFunction = lossFunction;
        this.iterations = iterations;
        this.cvState = new ConversionState(conversionCheck, convergenceRate);

        OptimizerOptions.propcessOptions(cl, optimizerOptions);

        return cl;
    }

    public enum FeatureType {
        STRING, INT, LONG
    }

    @Nonnull
    private static FeatureType getFeatureType(@Nonnull ListObjectInspector featureListOI)
            throws UDFArgumentException {
        final ObjectInspector featureOI = featureListOI.getListElementObjectInspector();
        if (featureOI instanceof StringObjectInspector) {
            return FeatureType.STRING;
        } else if (featureOI instanceof IntObjectInspector) {
            return FeatureType.INT;
        } else if (featureOI instanceof LongObjectInspector) {
            return FeatureType.LONG;
        } else {
            throw new UDFArgumentException("Feature object inspector must be one of "
                    + "[StringObjectInspector, IntObjectInspector, LongObjectInspector]: "
                    + featureOI.toString());
        }
    }

    @Nonnull
    protected final ObjectInspector getFeatureOutputOI(@Nonnull final FeatureType featureType)
            throws UDFArgumentException {
        final PrimitiveObjectInspector outputOI;
        if (dense_model) {
            // TODO validation
            outputOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector; // see DenseModel (long/string is also parsed as int)
        } else {
            switch (featureType) {
                case STRING:
                    outputOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                    break;
                case INT:
                    outputOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
                    break;
                case LONG:
                    outputOI = PrimitiveObjectInspectorFactory.javaLongObjectInspector;
                    break;
                default:
                    throw new IllegalStateException("Unexpected feature type: " + featureType);
            }
        }
        return outputOI;
    }

    @Nonnull
    protected StructObjectInspector getReturnOI(@Nonnull ObjectInspector featureOutputOI) {
        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

        fieldNames.add("feature");
        fieldOIs.add(featureOutputOI);
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

    protected void recordTrainSampleToTempFile(@Nonnull final FeatureValue[] featureVector,
            final float target) throws HiveException {
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
            this.inputBuf = buf = ByteBuffer.allocateDirect(1024 * 1024); // 1 MB
            this.fileIO = dst = new NioStatefullSegment(file, false);
        }

        int featureVectorBytes = 0;
        for (FeatureValue f : featureVector) {
            if (f == null) {
                continue;
            }
            int featureLength = f.getFeatureAsString().length();

            // feature as String (even if it is Text or Integer)
            featureVectorBytes += SizeOf.CHAR * featureLength;

            // NIOUtils.putString() first puts the length of string before string itself
            featureVectorBytes += SizeOf.INT;

            // value
            featureVectorBytes += SizeOf.DOUBLE;
        }

        // feature length, feature 1, feature 2, ..., feature n, target
        int recordBytes = SizeOf.INT + featureVectorBytes + SizeOf.FLOAT;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(featureVector.length);
        for (FeatureValue f : featureVector) {
            writeFeatureValue(buf, f);
        }
        buf.putFloat(target);
    }

    private static void writeFeatureValue(@Nonnull final ByteBuffer buf,
            @Nonnull final FeatureValue f) {
        NIOUtils.putString(f.getFeatureAsString(), buf);
        buf.putDouble(f.getValue());
    }

    @Nonnull
    private static FeatureValue readFeatureValue(@Nonnull final ByteBuffer buf,
            @Nonnull final FeatureType featureType) {
        final String featureStr = NIOUtils.getString(buf);
        final Object feature;
        switch (featureType) {
            case STRING:
                feature = featureStr;
                break;
            case INT:
                feature = Integer.valueOf(featureStr);
                break;
            case LONG:
                feature = Long.valueOf(featureStr);
                break;
            default:
                throw new IllegalStateException("Unexpected feature type " + featureType
                        + " for feature: " + featureStr);
        }
        double value = buf.getDouble();
        return new FeatureValue(feature, value);
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
            if (featureType == FeatureType.STRING) {
                String s = f.toString();
                fv = FeatureValue.parseFeatureAsString(s);
            } else {
                Object k = ObjectInspectorUtils.copyToStandardObject(f, featureInspector,
                    ObjectInspectorCopyOption.JAVA); // should be Integer or Long
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
        float loss = lossFunction.loss(predicted, target);
        cvState.incrLoss(loss); // retain cumulative loss to check convergence

        final float dloss = lossFunction.dloss(predicted, target);
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
        finalizeTraining();
        forwardModel();
        this.accumulated = null;
        this.model = null;
    }

    @VisibleForTesting
    public void finalizeTraining() throws HiveException {
        if (count == 0L) {
            this.model = null;
            return;
        }
        if (is_mini_batch) { // Update model with accumulated delta
            batchUpdate();
        }
        if (iterations > 1) {
            runIterativeTraining(iterations);
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

                for (int iter = 2; iter <= iterations; iter++) {
                    cvState.next();
                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        int featureVectorLength = buf.getInt();
                        final FeatureValue[] featureVector = new FeatureValue[featureVectorLength];
                        for (int j = 0; j < featureVectorLength; j++) {
                            featureVector[j] = readFeatureValue(buf, featureType);
                        }
                        float target = buf.getFloat();
                        train(featureVector, target);
                    }
                    buf.rewind();

                    if (is_mini_batch) { // Update model with accumulated delta
                        batchUpdate();
                    }

                    if (cvState.isConverged(numTrainingExamples)) {
                        break;
                    }
                }
                logger.info("Performed "
                        + cvState.getCurrentIteration()
                        + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on memory (thus "
                        + NumberUtils.formatNumber(numTrainingExamples
                                * cvState.getCurrentIteration()) + " training updates in total) ");
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
                for (int iter = 2; iter <= iterations; iter++) {
                    cvState.next();
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
                                featureVector[j] = readFeatureValue(buf, featureType);
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

                    if (cvState.isConverged(numTrainingExamples)) {
                        break;
                    }
                }
                logger.info("Performed "
                        + cvState.getCurrentIteration()
                        + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on a secondary storage (thus "
                        + NumberUtils.formatNumber(numTrainingExamples
                                * cvState.getCurrentIteration()) + " training updates in total)");
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
    public double getCumulativeLoss() {
        return (cvState == null) ? Double.NaN : cvState.getCumulativeLoss();
    }
}
