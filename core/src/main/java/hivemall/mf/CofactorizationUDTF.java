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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NioStatefulSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.*;


/**
 * Cofactorization for implicit and explicit recommendation
 */
@Description(name = "train_cofactor",
        value = "_FUNC_(string context, array<string> features, boolean is_validation, boolean is_item, array<string> sppmi [, String options])"
                + " - Returns a relation <string context, array<float> theta, array<floamft> beta>")
public class CofactorizationUDTF extends UDTFWithOptions {
    private static final Log LOG = LogFactory.getLog(CofactorizationUDTF.class);

    // Option variables
    // The number of latent factors
    private int factor;
    // The scaling hyperparameter for zero entries in the rank matrix
    private float c0;
    // The scaling hyperparameter for non-zero entries in the rank matrix
    private float c1;
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

    // validation metric
    private ValidationMetric validationMetric;

    // Initialization strategy of rank matrix
    private CofactorModel.RankInitScheme rankInit;

    // Model itself
    @VisibleForTesting
    protected CofactorModel model;

    // Variable managing status of learning
    private ConversionState validationState;
    private int numValPerRecord;

    // Input OIs and Context
    private StringObjectInspector userOI;
    @VisibleForTesting
    protected StringObjectInspector itemOI;

    private BooleanObjectInspector isValidationOI;
    @VisibleForTesting
    protected ListObjectInspector sppmiOI;

    // Used for iterations
    @VisibleForTesting
    protected NioStatefulSegment fileIO;
    private ByteBuffer inputBuf;
    protected long numValidations;
    protected long numTraining;

    // training data
    private Map<String, List<String>> userToItems;
    private Map<String, List<String>> itemToUsers;
    private Map<String, Feature[]> sppmi;

    // validation
    private Random rand;
    private double validationRatio;
    private List<String> validationUsers;
    private List<String> validationItems;

    static class MiniBatch {
        private List<TrainingSample> users;
        private List<TrainingSample> items;
        private List<TrainingSample> validationSamples;

        protected MiniBatch() {
            users = new ArrayList<>();
            items = new ArrayList<>();
            validationSamples = new ArrayList<>();
        }

        protected void add(TrainingSample sample) {
            if (sample.isValidation) {
                validationSamples.add(sample);
            } else {
                if (sample.isItem()) {
                    items.add(sample);
                } else {
                    users.add(sample);
                }
            }
        }

        protected void clear() {
            users.clear();
            items.clear();
            validationSamples.clear();
        }

        protected int trainingSize() {
            return items.size() + users.size();
        }

        protected int validationSize() {
            return validationSamples.size();
        }

        protected List<TrainingSample> getItems() {
            return items;
        }

        protected List<TrainingSample> getUsers() {
            return users;
        }

        public List<TrainingSample> getValidationSamples() {
            return validationSamples;
        }
    }

    static class TrainingSample {
        protected String context;
        protected Feature[] features;
        protected Feature[] sppmi;
        protected boolean isValidation;

        protected TrainingSample(@Nonnull final String context, @Nonnull final Feature[] features, final boolean isValidation, @Nullable final Feature[] sppmi) {
            this.context = context;
            this.features = features;
            this.sppmi = sppmi;
            this.isValidation = isValidation;
        }

        protected boolean isItem() {
            return sppmi != null;
        }
    }

    enum ValidationMetric {
        AUC, OBJECTIVE;

        static ValidationMetric resolve(@Nonnull final String opt) {
            switch (opt.toLowerCase()) {
                case "auc":
                    return AUC;
                case "objective":
                case "loss":
                    return OBJECTIVE;
                default:
                    throw new IllegalArgumentException(opt + " is not a supported Validation Metric.");
            }
        }
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k", "factor", true, "The number of latent factor [default: 10] "
                + " Note this is alias for `factors` option.");
        opts.addOption("f", "factors", true, "The number of latent factor [default: 10]");
        opts.addOption("lt", "lambda_theta", true, "The theta regularization factor [default: 1e-5]");
        opts.addOption("lb", "lambda_beta", true, "The beta regularization factor [default: 1e-5]");
        opts.addOption("lg", "lambda_gamma", true, "The gamma regularization factor [default: 1.0]");
        opts.addOption("c0", "c0", true,
                "The scaling hyperparameter for zero entries in the rank matrix [default: 0.1]");
        opts.addOption("c1", "c1", true,
                "The scaling hyperparameter for non-zero entries in the rank matrix [default: 1.0]");
        opts.addOption("gb", "global_bias", true, "The global bias [default: 0.0]");
        opts.addOption("update_gb", "update_global_bias", true,
                "Whether update (and return) the global bias or not [default: false]");
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
        opts.addOption("disable_bias", "no_bias", false, "Turn off bias clause");
        // normalization
        opts.addOption("disable_norm", "disable_l2norm", false, "Disable instance-wise L2 normalization");
        // validation
        opts.addOption("disable_cv", "disable_cvtest", false, "Whether to disable convergence check [default: enabled]");
        opts.addOption("cv_rate", "convergence_rate", true, "Threshold to determine convergence [default: 0.005]");
        opts.addOption("val_metric", "validation_metric", true, "Metric to use for validation ['auc', 'objective']");
        opts.addOption("val_ratio", "validation_ratio", true, "Proportion of examples to use as validation data [default: 0.125]");
        opts.addOption("num_val", "num_validation_examples_per_record", true, "Number of validation examples to use per record [default: 10]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        String rankInitOpt = "gaussian";
        float maxInitValue = 1.f;
        double initStdDev = 0.01d;
        boolean convergenceCheck = true;
        double convergenceRate = 0.005d;
        String validationMetricOpt = "auc";
        this.c0 = 0.1f;
        this.c1 = 1.0f;
        this.lambdaTheta = 1e-5f;
        this.lambdaBeta = 1e-5f;
        this.lambdaGamma = 1.0f;
        this.globalBias = 0.f;
        this.maxIters = 1;
        this.factor = 10;
        this.numValPerRecord = 10;
        this.validationRatio = 0.125;

        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);
            if (cl.hasOption("factors")) {
                this.factor = Primitives.parseInt(cl.getOptionValue("factors"), factor);
            } else {
                this.factor = Primitives.parseInt(cl.getOptionValue("factor"), factor);
            }
            this.lambdaTheta = Primitives.parseFloat(cl.getOptionValue("lambda_theta"), lambdaTheta);
            this.lambdaBeta = Primitives.parseFloat(cl.getOptionValue("lambda_beta"), lambdaBeta);
            this.lambdaGamma = Primitives.parseFloat(cl.getOptionValue("lambda_gamma"), lambdaGamma);

            this.c0 = Primitives.parseFloat(cl.getOptionValue("c0"), c0);
            this.c1 = Primitives.parseFloat(cl.getOptionValue("c1"), c1);

            this.globalBias = Primitives.parseFloat(cl.getOptionValue("global_bias"), globalBias);
            this.updateGlobalBias = cl.hasOption("update_global_bias");

            rankInitOpt = cl.getOptionValue("rankinit", rankInitOpt);
            maxInitValue = Primitives.parseFloat(cl.getOptionValue("max_init_value"), maxInitValue);
            initStdDev = Primitives.parseDouble(cl.getOptionValue("min_init_stddev"), initStdDev);

            if (cl.hasOption("iter")) {
                this.maxIters = Primitives.parseInt(cl.getOptionValue("iter"), maxIters);
            } else {
                this.maxIters = Primitives.parseInt(cl.getOptionValue("max_iters"), maxIters);
            }
            if (maxIters < 1) {
                throw new UDFArgumentException(
                        "'-max_iters' must be greater than or equal to 1: " + maxIters);
            }

            convergenceCheck = !cl.hasOption("disable_cvtest");
            convergenceRate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), convergenceRate);
            validationMetricOpt = cl.getOptionValue("validation_metric", validationMetricOpt);
            this.numValPerRecord = Primitives.parseInt(cl.getOptionValue("num_validation_examples_per_record"), numValPerRecord);
            this.validationRatio = Primitives.parseDouble(cl.getOptionValue("validation_ratio"), this.validationRatio);
            if (this.validationRatio > 1 || this.validationRatio < 0) {
                throw new UDFArgumentException(
                        "'-validation_ratio' must be between 0.0 and 1.0"
                );
            }
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
        rankInit.setInitStdDev(initStdDev);
        this.validationState = new ConversionState(convergenceCheck, convergenceRate);
        this.validationMetric = ValidationMetric.resolve(validationMetricOpt);
        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 3) {
            throw new UDFArgumentException(
                    "_FUNC_ takes 3 arguments: string user, string item, array<string> sppmi [, CONSTANT STRING options]");
        }
        this.userOI = HiveUtils.asStringOI(argOIs[0]);
        this.itemOI = HiveUtils.asStringOI(argOIs[1]);
        this.sppmiOI = HiveUtils.asListOI(argOIs[2]);
        HiveUtils.validateFeatureOI(sppmiOI.getListElementObjectInspector());

        processOptions(argOIs);

        this.model = new CofactorModel(factor, rankInit, c0, c1, lambdaTheta, lambdaBeta, lambdaGamma, globalBias,
                validationMetric, numValPerRecord, LOG);

        userToItems = new HashMap<>();
        itemToUsers = new HashMap<>();

        validationUsers = new ArrayList<>();
        validationItems = new ArrayList<>();

        rand = new Random(31);

        List<String> fieldNames = new ArrayList<String>();
        List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("context");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("theta");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableFloatObjectInspector));
        fieldNames.add("beta");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableFloatObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        final String user = userOI.getPrimitiveJavaObject(args[0]);
        final String item = itemOI.getPrimitiveJavaObject(args[1]);
        final Feature[] sppmiVec = Feature.parseFeatures(args[2], sppmiOI, null, false);
        if (sppmiVec == null) {
            throw new HiveException("sppmi value is null");
        }
        recordSample(user, item, sppmiVec);
    }

    private static void addToMap(@Nonnull final Map<String, List<String>> map, @Nonnull final String key, @Nonnull final String value) {
        List<String> values = map.get(key);
        final boolean isNewKey = values == null;
        if (isNewKey) {
            values = new ArrayList<>();
            values.add(value);
            map.put(key, values);
        } else {
            values.add(value);
        }
    }

    private void recordSample(@Nonnull final String user, @Nonnull final String item, @Nonnull final Feature[] sppmiVec) {
        // validation data
        if (rand.nextDouble() < validationRatio) {
            addValidationSample(user, item);
        } else {
            // train
            addToMap(userToItems, user, item);
            addToMap(itemToUsers, item, user);
        }
        addToSPPMI(item, sppmiVec);
    }

    private void addValidationSample(@Nonnull final String user, @Nonnull final String item) {
        validationUsers.add(user);
        validationItems.add(item);
    }

    private void addToSPPMI(@Nonnull final String item, @Nonnull final Feature[] sppmiVec) {
        if (sppmi.containsKey(item)) {
            return;
        }
        sppmi.put(item, sppmiVec);
    }

    @Override
    public void close() throws HiveException {
        try {
            model.registerUsers(userToItems.keySet());
            model.registerItems(itemToUsers.keySet());

            final Reporter reporter = getReporter();
            final Counters.Counter iterCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "iteration");

            final Counters.Counter userCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "users");
            final Counters.Counter itemCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "items");
            final Counters.Counter skippedUserCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "skippedUsers");
            final Counters.Counter skippedItemCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "skippedItems");

            final Counters.Counter thetaTotalCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "thetaTotalFeaturesCounter");
            final Counters.Counter thetaTrainableCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "thetaTrainableFeaturesCounter");

            final Counters.Counter betaTotalCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "betaTotalFeaturesCounter");
            final Counters.Counter betaTrainableCounter = (reporter == null) ? null
                    : reporter.getCounter("hivemall.mf.Cofactor$Counter", "betaTrainableFeaturesCounter");

            model.registerCounters(userCounter, itemCounter, skippedUserCounter, skippedItemCounter, thetaTrainableCounter,
                    thetaTotalCounter, betaTrainableCounter, betaTotalCounter);

            if (LOG.isInfoEnabled()) {
                File tmpFile = fileIO.getFile();
                LOG.info("Wrote " + numTraining
                        + " records to a temporary file for iterative training: "
                        + tmpFile.getAbsolutePath() + " (" + FileUtils.prettyFileSize(tmpFile)
                        + ")");
            }

            for (int iteration = 0; iteration < maxIters; iteration++) {
                // train the model on a full batch (i.e., all the data) using mini-batch updates
                validationState.next();
                reportProgress(reporter);
                setCounterValue(iterCounter, iteration);
                runTrainingIteration();

                System.out.println("Validation loss: " + validationState.getAverageLoss(numValidations));

                LOG.info("Performed " + iteration + " iterations of "
                        + NumberUtils.formatNumber(maxIters) + " with " + numTraining + " training examples and "
                        + numValidations + " validation examples.");
//                        + " training examples on a secondary storage (thus "
//                        + NumberUtils.formatNumber(_t) + " training updates in total), used "
//                        + _numValidations + " validation examples");

                if (validationState.isConverged(numTraining)) {
                    break;
                }
            }
            forwardModel();
        } finally {
            this.model = null;
        }
    }

    private void forwardModel() throws HiveException {
        if (model == null) {
            return;
        }

        final Text context = new Text();
        final FloatWritable[] theta = HiveUtils.newFloatArray(factor, 0.f);
        final FloatWritable[] beta = HiveUtils.newFloatArray(factor, 0.f);
        final Object[] forwardObj = new Object[] {context, theta, null};

        int numUsersForwarded = 0, numItemsForwarded = 0;

        for (Map.Entry<String, double[]> entry : model.getTheta().entrySet()) {
            context.set(entry.getKey());
            copyTo(entry.getValue(), theta);
            forward(forwardObj);
            numUsersForwarded++;
        }

        forwardObj[1] = null;
        forwardObj[2] = beta;
        for (Map.Entry<String, double[]> entry : model.getBeta().entrySet()) {
            context.set(entry.getKey());
            copyTo(entry.getValue(), beta);
            forward(forwardObj);
            numItemsForwarded++;
        }
        LOG.info("Forwarded the prediction model of " + numUsersForwarded + " user rows (theta) and " + numItemsForwarded + " item rows (beta).]");

    }

    private void copyTo(@Nonnull final double[] src, @Nonnull final FloatWritable[] dst) {
        for (int k = 0, size = factor; k < size; k++) {
            dst[k].set((float) src[k]);
        }
    }

    private void runTrainingIteration() throws HiveException {
        model.updateWithUsers(userToItems);
        model.updateWithItems(itemToUsers, sppmi);
//        model.validate()
    }

    private void validate(List<TrainingSample> samples) throws HiveException {
        for (TrainingSample sample : samples) {
            final Double loss = model.validate(sample, 31);
            if (loss != null) {
                if (!NumberUtils.isFinite(loss)) {
                    throw new HiveException("Non-finite validation loss encountered");
                }
                validationState.incrLoss(loss);
            }
        }
    }
}
