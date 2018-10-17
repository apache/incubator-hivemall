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
import hivemall.common.ConversionState;
import hivemall.fm.Feature;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.NioStatefulSegment;
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
    private static final Log logger = LogFactory.getLog(CofactorizationUDTF.class);
    private static final int RECORD_BYTES = SizeOf.INT + SizeOf.INT + SizeOf.FLOAT;
    private static final int REQUIRED_BYTES = SizeOf.INT + RECORD_BYTES;

    // Option variables
    /** The number of latent factors */
    protected int factor;
    /** The scaling hyperparameter for zero entries in the rank matrix */
    protected float scale_zero;
    /** The scaling hyperparameter for non-zero entries in the rank matrix */
    protected float scale_nonzero;
    /** The preferred size of the miniBatch for training */
    protected int batchSize;
    /** The initial mean rating */
    protected float globalBias;
    /** Whether update (and return) the mean rating or not */
    protected boolean updateGlobalBias;
    /** The number of iterations */
    protected int iterations;
    /** Whether to use bias clause */
    protected boolean useBiasClause;
    /** Whether to use normalization */
    protected boolean useL2Norm;
    /** Whether to parse feature as integer */
    protected boolean parseFeatureAsInt;
    /** regularization hyperparameters */
    protected float lambdaTheta;
    protected float lambdaBeta;
    protected float lambdaGamma;

    /** Initialization strategy of rank matrix */
    protected CofactorModel.RankInitScheme rankInit;

    // Model itself
    protected CofactorModel model;
    protected int numItems;

    // Variable managing status of learning
    /** The number of processed training examples */
    protected long count;
    protected ConversionState cvState;

    // Input OIs and Context
    protected StringObjectInspector parentOI;
    protected ListObjectInspector childrenOI;
    protected BooleanObjectInspector isParentAnItemOI;
    protected ListObjectInspector sppmiVectorOI;

    // Used for iterations
    protected NioStatefulSegment fileIO;
    protected ByteBuffer inputBuf;
    private long lastWritePos;

    private Feature parentProbe;
    private Feature[] childrenProbe;
    private Feature[] sppmiVectorProbe;
    private boolean isParentAnItemProbe;
    private int numValidations;
    private int numTraining;
    private MiniBatch miniBatch;

    static class MiniBatch {
        protected int maxSize;
        private List<TrainingSample> users;
        private List<TrainingSample> items;

        protected MiniBatch(int maxSize) {
            this.maxSize = maxSize;
        }

        protected void add(TrainingSample sample) {
            if (size() == this.maxSize) {
                return;
            }

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
        protected Feature parent;
        protected Feature[] children;
        protected Feature[] sppmiVector;

        protected TrainingSample(Feature parent, Feature[] children, Feature[] sppmiVector) {
            this.parent = parent;
            this.children = children;
            this.sppmiVector = sppmiVector;
        }

        protected boolean isItem() {
            return sppmiVector != null;
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
        opts.addOption("n", "num_items", false, "Number of items");
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
        opts.addOption("disable_cv", "disable_cvtest", false,
                "Whether to disable convergence check [default: enabled]");
        opts.addOption("cv_rate", "convergence_rate", true,
                "Threshold to determine convergence [default: 0.005]");
        opts.addOption("disable_bias", "no_bias", false, "Turn off bias clause");
        // feature representation
        opts.addOption("int_feature", "feature_as_integer", false,
                "Parse a feature as integer [default: OFF]");
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

        if (argOIs.length >= 4) {
            String rawArgs = HiveUtils.getConstString(argOIs[3]);
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
            this.batchSize = Primitives.parseInt(cl.getOptionValue("batch_size"), 1024);
            if (cl.hasOption("num_items")) {
                this.numItems = Primitives.parseInt(cl.getOptionValue("num_items"), 1024);
            } else {
                throw new UDFArgumentException("-num_items must be specified");
            }
            this.globalBias = Primitives.parseFloat(cl.getOptionValue("gb"), 0.f);
            this.updateGlobalBias = cl.hasOption("update_gb");
            rankInitOpt = cl.getOptionValue("rankinit");
            maxInitValue = Primitives.parseFloat(cl.getOptionValue("max_init_value"), 1.f);
            initStdDev = Primitives.parseDouble(cl.getOptionValue("min_init_stddev"), 0.01d);
            if (cl.hasOption("iter")) {
                this.iterations = Primitives.parseInt(cl.getOptionValue("iter"), 1);
            } else {
                this.iterations = Primitives.parseInt(cl.getOptionValue("iterations"), 1);
            }
            if (iterations < 1) {
                throw new UDFArgumentException(
                        "'-iterations' must be greater than or equal to 1: " + iterations);
            }
            conversionCheck = !cl.hasOption("disable_cvtest");
            convergenceRate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), convergenceRate);
            boolean noBias = cl.hasOption("no_bias");
            this.useBiasClause = !noBias;
            if (noBias && updateGlobalBias) {
                throw new UDFArgumentException(
                        "Cannot set both `update_gb` and `no_bias` option");
            }
            this.parseFeatureAsInt = cl.hasOption("int_feature");
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
                    "_FUNC_ takes 3 arguments: array<string> x, array<string> sppmiVector [, CONSTANT STRING options]");
        }
        this.parentOI = HiveUtils.asStringOI(argOIs[0]);
        this.childrenOI = HiveUtils.asListOI(argOIs[1]);
        HiveUtils.validateFeatureOI(childrenOI.getListElementObjectInspector());
        this.isParentAnItemOI = HiveUtils.asBooleanOI(argOIs[2]);
        this.sppmiVectorOI = HiveUtils.asListOI(argOIs[3]);
        HiveUtils.validateFeatureOI(sppmiVectorOI.getListElementObjectInspector());

        processOptions(argOIs);

        this.model = new CofactorModel(factor, rankInit, scale_zero, scale_nonzero, lambdaTheta, lambdaBeta, lambdaGamma);
        this.miniBatch = new MiniBatch(this.batchSize);
        this.count = 0L;
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
        if (useBiasClause) {
            fieldNames.add("Bu");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
            fieldNames.add("Bi");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
            fieldNames.add("Bc");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
            if (updateGlobalBias) {
                fieldNames.add("mu");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);
            }
        }
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (args.length != 3) {
            throw new HiveException("should have 3 args, but have " + args.length);
        }

        String parentString = parentOI.getPrimitiveJavaObject(args[0]);
        Feature parent = Feature.parseFeature(parentString,false);
        assert parent != null;

        Feature[] children = parseFeatures(args[1], childrenOI, childrenProbe);
        assert children != null;

        Boolean isParentAnItem = isParentAnItemOI.get(args[2]);
        Feature[] sppmiVector = null;
        if (isParentAnItem) {
             sppmiVector = parseFeatures(args[2], sppmiVectorOI, sppmiVectorProbe);
        }

        model.recordAsParent(parent, isParentAnItem);

        this.parentProbe = parent;
        this.childrenProbe = children;
        this.isParentAnItemProbe = isParentAnItem;
        this.sppmiVectorProbe = sppmiVector;

        addToBatch(parent, children, sppmiVector);
        recordTrain(parent, children, sppmiVector);
        trainMiniBatch();
    }

    @Nullable
    protected Feature[] parseFeatures(@Nonnull final Object arg, ListObjectInspector listOI, Feature[] probe) throws HiveException {
        Feature[] rawFeatures = Feature.parseFeatures(arg, listOI, probe, false);
        Feature[] nnzFeatures = createNnzFeatureArray(rawFeatures);
        return nnzFeatures;
    }

    protected Feature[] createNnzFeatureArray(Feature[] x) {
        int nnz = countNnzFeatures(x);
        Feature[] nnzFeatures = new Feature[nnz];
        int i = 0;
        for (Feature f: x) {
            if (f.getValue() != 0.d) {
                nnzFeatures[i++] = f;
            }
        }
        return nnzFeatures;
    }

    private int countNnzFeatures(Feature[] x) {
        int nnz = 0;
        for (Feature f : x) {
            if (f.getValue() != 0.d) {
                nnz++;
            }
        }
        return nnz;
    }

    private void trainMiniBatch() {
        if (miniBatch.size() < batchSize) {
            return;
        }
        numTraining += miniBatch.size();
        model.updateWithUsers(miniBatch.getUsers());
        model.updateWithItems(miniBatch.getItems());
        miniBatch.clear();
    }

    private void addToBatch(final Feature parent, final Feature[] children, final Feature[] sppmiVector) {
        TrainingSample sample = new TrainingSample(parent, children, sppmiVector);
        miniBatch.add(sample);
    }

    private void recordTrain(final Feature parent, final Feature[] children, final Feature[] sppmiVector)
            throws HiveException {
        if (iterations <= 1) {
            return;
        }

        ByteBuffer inputBuf = this.inputBuf;
        NioStatefulSegment dst = this.fileIO;
        if (inputBuf == null) {
            final File file;
            try {
                file = File.createTempFile("hivemall_cofactor", ".sgmt");
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException(
                            "Cannot write a temporary file: " + file.getAbsolutePath());
                }
                logger.info("Record training examples to a file: " + file.getAbsolutePath());
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            } catch (Throwable e) {
                throw new UDFArgumentException(e);
            }

            this.inputBuf = inputBuf = ByteBuffer.allocateDirect(1024 * 1024); // 1 MiB
            this.fileIO = dst = new NioStatefulSegment(file, false);
        }

        int parentBytes = parent.bytes();
        int childrenBytes = Feature.requiredBytes(children);
        int isParentAnItemBytes = SizeOf.BYTE;
        int sppmiVectorBytes = sppmiVector != null ? Feature.requiredBytes(sppmiVector) : 0;

        int recordBytes = parentBytes + childrenBytes + isParentAnItemBytes + sppmiVectorBytes;
        int requiredBytes = SizeOf.INT + recordBytes;
        int remain = inputBuf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(inputBuf, dst);
        }

        inputBuf.putInt(recordBytes);
        parent.writeTo(inputBuf);
        writeFeaturesToBuffer(children, inputBuf);
        if (sppmiVector != null) {
            inputBuf.put(TRUE_BYTE);
            writeFeaturesToBuffer(sppmiVector, inputBuf);
        } else {
            inputBuf.put(FALSE_BYTE);
        }
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
        if (!batch.isEmpty()) {
            trainBatch();
        }

        // train for t iterations
    }
}
