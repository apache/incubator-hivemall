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
package hivemall.mf;

import hivemall.UDTFWithOptions;
import hivemall.annotations.VisibleForTesting;
import hivemall.common.ConversionState;
import hivemall.fm.Feature;
import hivemall.fm.StringFeature;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NIOUtils;
import hivemall.utils.io.NioStatefulSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static hivemall.utils.lang.Primitives.FALSE_BYTE;
import static hivemall.utils.lang.Primitives.TRUE_BYTE;

public class CofactorizationUDTF extends UDTFWithOptions {
    private static final Log LOG = LogFactory.getLog(CofactorizationUDTF.class);

    // Option variables
    // The number of latent factors
    private int factor;
    // The scaling hyperparameter for zero entries in the rank matrix
    private float scale_zero;
    // The scaling hyperparameter for non-zero entries in the rank matrix
    private float scale_nonzero;
    // The initial mean rating
    private float globalBias;
    // Whether update (and return) the mean rating or not
    private boolean updateGlobalBias;
    // The number of iterations
    private int maxIters;
    // Whether to use bias clause
    private boolean useBiasClause;
    // Whether to use normalization
    private boolean useL2Norm;
    // regularization hyperparameters
    private float lambdaTheta;
    private float lambdaBeta;
    private float lambdaGamma;

    // Initialization strategy of rank matrix
    private CofactorModel.RankInitScheme rankInit;

    // Model itself
    @VisibleForTesting
    protected CofactorModel model;

    // Variable managing status of learning
    private ConversionState cvState;
    private ConversionState validationState;

    // Input OIs and Context
    private StringObjectInspector contextOI;
    @VisibleForTesting
    protected ListObjectInspector featuresOI;
    private BooleanObjectInspector isItemOI;
    @VisibleForTesting
    protected ListObjectInspector sppmiOI;

    // Used for iterations
    @VisibleForTesting
    protected NioStatefulSegment fileIO;
    private ByteBuffer inputBuf;
    private long lastWritePos;

    private String contextProbe;
    private Feature[] featuresProbe;
    private Feature[] sppmiProbe;
    private boolean isItemProbe;
    private long numValidations;
    private long numTraining;


    static class MiniBatch {
        private List<TrainingSample> users;
        private List<TrainingSample> items;

        protected MiniBatch() {
            users = new ArrayList<>();
            items = new ArrayList<>();
        }

        protected void add(TrainingSample sample) {
            if (sample.isItem()) {
                items.add(sample);
            } else {
                users.add(sample);
            }
        }

        protected void clear() {
            users.clear();
            items.clear();
        }

        protected int size() {
            return items.size() + users.size();
        }

        protected List<TrainingSample> getItems() {
            return items;
        }

        protected List<TrainingSample> getUsers() {
            return users;
        }
    }

    static class TrainingSample {
        protected String context;
        protected Feature[] features;
        protected Feature[] sppmi;

        protected TrainingSample(String context, Feature[] features, Feature[] sppmi) {
            this.context = context;
            this.features = features;
            this.sppmi = sppmi;
        }

        protected boolean isItem() {
            return sppmi != null;
        }
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k", "factor", true, "The number of latent factor [default: 10] "
                + " Note this is alias for `factors` option.");
        opts.addOption("f", "factors", true, "The number of latent factor [default: 10]");
        opts.addOption("r", "lambda", true, "The regularization factor [default: 0.03]");
        opts.addOption("c0", "scale_zero", true,
                "The scaling hyperparameter for zero entries in the rank matrix [default: 0.1]");
        opts.addOption("c1", "scale_nonzero", true,
                "The scaling hyperparameter for non-zero entries in the rank matrix [default: 1.0]");
        opts.addOption("b", "batch_size", true, "The miniBatch size for training [default: 1024]");
        opts.addOption("gb", "global_bias", true, "The global bias [default: 0.0]");
        opts.addOption("update_gb", "update_gb", false,
                "Whether update (and return) the global bias or not");
        opts.addOption("rankinit", true,
                "Initialization strategy of rank matrix [random, gaussian] (default: gaussian)");
        opts.addOption("maxval", "max_init_value", true,
                "The maximum initial value in the rank matrix [default: 1.0]");
        opts.addOption("min_init_stddev", true,
                "The minimum standard deviation of initial rank matrix [default: 0.01]");
        opts.addOption("iters", "iterations", true, "The number of iterations [default: 1]");
        opts.addOption("iter", true,
                "The number of iterations [default: 1] Alias for `-iterations`");
        opts.addOption("max_iters", "max_iters", true, "The number of iterations [default: 1]");
        opts.addOption("disable_cv", "disable_cvtest", false,
                "Whether to disable convergence check [default: enabled]");
        opts.addOption("cv_rate", "convergence_rate", true,
                "Threshold to determine convergence [default: 0.005]");
        opts.addOption("disable_bias", "no_bias", false, "Turn off bias clause");
        // normalization
        opts.addOption("disable_norm", "disable_l2norm", false, "Disable instance-wise L2 normalization");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        String rankInitOpt = "gaussian";
        float maxInitValue = 1.f;
        double initStdDev = 0.1d;
        boolean conversionCheck = true;
        double convergenceRate = 0.005d;

        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);
            if (cl.hasOption("factors")) {
                this.factor = Primitives.parseInt(cl.getOptionValue("factors"), 10);
            } else {
                this.factor = Primitives.parseInt(cl.getOptionValue("factor"), 10);
            }
            this.lambdaTheta = Primitives.parseFloat(cl.getOptionValue("lambda_theta"), 1e-5f);
            this.lambdaBeta = Primitives.parseFloat(cl.getOptionValue("lambda_beta"), 1e-5f);
            this.lambdaGamma = Primitives.parseFloat(cl.getOptionValue("lambda_gamma"), 1e+0f);
            this.scale_zero = Primitives.parseFloat(cl.getOptionValue("scale_zero"), 0.1f);
            this.scale_nonzero = Primitives.parseFloat(cl.getOptionValue("scale_nonzero"), 1.0f);
            this.globalBias = Primitives.parseFloat(cl.getOptionValue("gb"), 0.f);
            this.updateGlobalBias = cl.hasOption("update_gb");
            rankInitOpt = cl.getOptionValue("rankinit");
            maxInitValue = Primitives.parseFloat(cl.getOptionValue("max_init_value"), 1.f);
            initStdDev = Primitives.parseDouble(cl.getOptionValue("min_init_stddev"), 0.01d);
            if (cl.hasOption("iter")) {
                this.maxIters = Primitives.parseInt(cl.getOptionValue("iter"), 1);
            } else {
                this.maxIters = Primitives.parseInt(cl.getOptionValue("maxIters"), 1);
            }
            if (maxIters < 1) {
                throw new UDFArgumentException(
                        "'-maxIters' must be greater than or equal to 1: " + maxIters);
            }
            conversionCheck = !cl.hasOption("disable_cvtest");
            convergenceRate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), convergenceRate);
            boolean noBias = cl.hasOption("no_bias");
            this.useBiasClause = !noBias;
            if (noBias && updateGlobalBias) {
                throw new UDFArgumentException(
                        "Cannot set both `update_gb` and `no_bias` option");
            }
            this.useL2Norm = !cl.hasOption("disable_l2norm");
        }
        this.rankInit = CofactorModel.RankInitScheme.resolve(rankInitOpt);
        rankInit.setMaxInitValue(maxInitValue);
        initStdDev = Math.max(initStdDev, 1.0d / factor);
        rankInit.setInitStdDev(initStdDev);
        this.cvState = new ConversionState(conversionCheck, convergenceRate);
        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 3) {
            throw new UDFArgumentException(
                    "_FUNC_ takes 5 arguments: string context, array<string> features, boolean is_item, array<string> sppmi [, CONSTANT STRING options]");
        }
        this.contextOI = HiveUtils.asStringOI(argOIs[0]);
        this.featuresOI = HiveUtils.asListOI(argOIs[1]);
        HiveUtils.validateFeatureOI(featuresOI.getListElementObjectInspector());
        this.isItemOI = HiveUtils.asBooleanOI(argOIs[2]);
        this.sppmiOI = HiveUtils.asListOI(argOIs[3]);
        HiveUtils.validateFeatureOI(sppmiOI.getListElementObjectInspector());

        processOptions(argOIs);

        this.model = new CofactorModel(factor, rankInit, scale_zero, scale_nonzero, lambdaTheta, lambdaBeta, lambdaGamma);
        this.lastWritePos = 0L;

        List<String> fieldNames = new ArrayList<String>();
        List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("idx");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("Pu");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableFloatObjectInspector));
        fieldNames.add("Qi");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableFloatObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (args.length != 4) {
            throw new HiveException("should have 4 args, but have " + args.length);
        }

        String context = contextOI.getPrimitiveJavaObject(args[0]);
        Feature[] features = parseFeatures(args[1], featuresOI, featuresProbe);
        if (features == null) {
            throw new HiveException("features must not be null");
        }

        Boolean isItem = isItemOI.get(args[2]);
        Feature[] sppmi = null;
        if (isItem) {
            sppmi = parseFeatures(args[3], sppmiOI, sppmiProbe);
        }

        model.recordContext(context, isItem);

        this.contextProbe = context;
        this.featuresProbe = features;
        this.isItemProbe = isItem;
        this.sppmiProbe = sppmi;

        recordTrain(context, features, sppmi);
    }

    @Nullable
    @VisibleForTesting
    protected static Feature[] parseFeatures(@Nullable final Object arg, ListObjectInspector listOI, Feature[] probe) throws HiveException {
        if (arg == null) {
            return null;
        }
        Feature[] rawFeatures = Feature.parseFeatures(arg, listOI, probe, false);
        return createNnzFeatureArray(rawFeatures);
    }

    @VisibleForTesting
    protected static Feature[] createNnzFeatureArray(@Nonnull Feature[] x) {
        int nnz = countNnzFeatures(x);
        Feature[] nnzFeatures = new Feature[nnz];
        int i = 0;
        for (Feature f : x) {
            if (f.getValue() != 0.d) {
                nnzFeatures[i++] = f;
            }
        }
        return nnzFeatures;
    }

    private static int countNnzFeatures(@Nonnull Feature[] x) {
        int nnz = 0;
        for (Feature f : x) {
            if (f.getValue() != 0.d) {
                nnz++;
            }
        }
        return nnz;
    }

    private Double trainMiniBatch(MiniBatch miniBatch) throws HiveException {
        model.updateWithUsers(miniBatch.getUsers());
        model.updateWithItems(miniBatch.getItems());
        return model.calculateLoss(miniBatch.getUsers(), miniBatch.getItems());
    }

    private void recordTrain(final String context, final Feature[] features, final Feature[] sppmi)
            throws HiveException {
        numTraining++;
        ByteBuffer inputBuf = this.inputBuf;
        NioStatefulSegment dst = this.fileIO;
        if (inputBuf == null) {
            final File file = createTempFile();
            this.inputBuf = inputBuf = ByteBuffer.allocateDirect(1024 * 1024); // 1 MiB
            this.fileIO = dst = new NioStatefulSegment(file, false);
        }

        writeRecordToBuffer(inputBuf, dst, context, features, sppmi);
    }

    private static void writeRecordToBuffer(@Nonnull final ByteBuffer inputBuf, @Nonnull final NioStatefulSegment dst, @Nonnull final String context,
                                            @Nonnull final Feature[] features, @Nullable final Feature[] sppmi) throws HiveException {
        int recordBytes = calculateRecordBytes(context, features, sppmi);
        int requiredBytes = SizeOf.INT + recordBytes;
        int remain = inputBuf.remaining();

        if (remain < requiredBytes) {
            writeBuffer(inputBuf, dst);
        }

        inputBuf.putInt(recordBytes);
        NIOUtils.putString(context, inputBuf);
        writeFeaturesToBuffer(features, inputBuf);
        if (sppmi != null) {
            inputBuf.put(TRUE_BYTE);
            writeFeaturesToBuffer(sppmi, inputBuf);
        } else {
            inputBuf.put(FALSE_BYTE);
        }
    }

    private static int calculateRecordBytes(String context, Feature[] features, Feature[] sppmi) {
        int contextBytes = SizeOf.INT + SizeOf.CHAR * context.length();
        int featuresBytes = SizeOf.INT + Feature.requiredBytes(features);
        int isItemBytes = SizeOf.BYTE;
        int sppmiBytes = sppmi != null ? SizeOf.INT + Feature.requiredBytes(sppmi) : 0;
        return contextBytes + featuresBytes + isItemBytes + sppmiBytes;
    }

    private static File createTempFile() throws UDFArgumentException {
        final File file;
        try {
            file = File.createTempFile("hivemall_cofactor", ".sgmt");
            file.deleteOnExit();
            if (!file.canWrite()) {
                throw new UDFArgumentException(
                        "Cannot write a temporary file: " + file.getAbsolutePath());
            }
            LOG.info("Record training examples to a file: " + file.getAbsolutePath());
        } catch (IOException ioe) {
            throw new UDFArgumentException(ioe);
        } catch (Throwable e) {
            throw new UDFArgumentException(e);
        }
        return file;
    }

    private static void writeFeaturesToBuffer(Feature[] features, ByteBuffer buffer) {
        buffer.putInt(features.length);
        for (Feature f : features) {
            f.writeTo(buffer);
        }
    }

    private static void writeBuffer(@Nonnull ByteBuffer srcBuf, @Nonnull NioStatefulSegment dst)
            throws HiveException {
        srcBuf.flip();
        try {
            dst.write(srcBuf);
        } catch (IOException e) {
            throw new HiveException("Exception causes while writing a buffer to file", e);
        }
        srcBuf.clear();
    }

    @Override
    public void close() throws HiveException {
        try {
            boolean lossIncreasedLastIter = false;

            final Reporter reporter = getReporter();
            final Counters.Counter iterCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "iteration");

            prepareForRead();

            if (LOG.isInfoEnabled()) {
                File tmpFile = fileIO.getFile();
                LOG.info("Wrote " + numTraining
                        + " records to a temporary file for iterative training: "
                        + tmpFile.getAbsolutePath() + " (" + FileUtils.prettyFileSize(tmpFile)
                        + ")");
            }

            for (int iteration = 0; iteration < maxIters; iteration++) {
                // train the model on a full batch (i.e., all the data) using mini-batch updates
//                validationState.next();
                cvState.next();
                reportProgress(reporter);
                setCounterValue(iterCounter, iteration);
                runTrainingIteration();

                LOG.info("Performed " + cvState.getCurrentIteration() + " iterations of "
                        + NumberUtils.formatNumber(maxIters));
//                        + " training examples on a secondary storage (thus "
//                        + NumberUtils.formatNumber(_t) + " training updates in total), used "
//                        + _numValidations + " validation examples");
            }
        } finally {
            // delete the temporary file and release resources
            try {
                fileIO.close(true);
            } catch (IOException e) {
                throw new HiveException(
                        "Failed to close a file: " + fileIO.getFile().getAbsolutePath(), e);
            }
            this.inputBuf = null;
            this.fileIO = null;
        }
    }

    @VisibleForTesting
    protected void prepareForRead() throws HiveException {
        // write training examples in buffer to a temporary file
        if (inputBuf.remaining() > 0) {
            writeBuffer(inputBuf, fileIO);
        }
        try {
            fileIO.flush();
        } catch (IOException e) {
            throw new HiveException(
                    "Failed to flush a file: " + fileIO.getFile().getAbsolutePath(), e);
        }
        fileIO.resetPosition();
    }

    private void runTrainingIteration() throws HiveException {
        fileIO.resetPosition();
        MiniBatch miniBatch = new MiniBatch();
        // read minibatch from disk into memory
        while (readMiniBatchFromFile(miniBatch)) {
            Double trainLoss = trainMiniBatch(miniBatch);
            if (trainLoss != null) {
                cvState.incrLoss(trainLoss);
            }
            miniBatch.clear();
        }

    }

    @Nonnull
    private static Feature instantiateFeature(@Nonnull final ByteBuffer input) {
        return new StringFeature(input);
    }

    @VisibleForTesting
    protected boolean readMiniBatchFromFile(MiniBatch miniBatch) throws HiveException {
        // writes training examples to a buffer in the temporary file
        final int bytesRead;
        try {
            bytesRead = fileIO.read(inputBuf);
        } catch (IOException e) {
            throw new HiveException(
                    "Failed to read a file: " + fileIO.getFile().getAbsolutePath(), e);
        }
        if (bytesRead == 0) { // reached file EOF
            return false;
        }

        // reads training examples from a buffer
        inputBuf.flip();
        int remain = inputBuf.remaining();
        if (remain < SizeOf.INT) {
            throw new HiveException("Illegal file format was detected");
        }
        while (remain >= SizeOf.INT) {
            int initialPos = inputBuf.position();
            int recordBytes = inputBuf.getInt();
            remain -= SizeOf.INT;

            if (remain < recordBytes) {
                // whole record can't fit in memory,
                // so end the mini-batch here
                inputBuf.position(initialPos);
                break;
            }
            final TrainingSample sample = readSampleFromBuffer(inputBuf);
            miniBatch.add(sample);
            remain -= recordBytes;
        }
        inputBuf.compact();
        return true;
    }

    private static TrainingSample readSampleFromBuffer(ByteBuffer inputBuf) {
        final String context = NIOUtils.getString(inputBuf);
        final Feature[] features = instantiateFeatureArray(inputBuf);
        final boolean isItem = (inputBuf.get() == TRUE_BYTE);
        final Feature[] sppmi = isItem ? instantiateFeatureArray(inputBuf) : null;
        return new TrainingSample(context, features, sppmi);
    }

    private static Feature[] instantiateFeatureArray(@Nonnull final ByteBuffer buffer) {
        final int numFeatures = buffer.getInt();
        final Feature[] features = new Feature[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            features[j] = instantiateFeature(buffer);
        }
        return features;
    }
}
