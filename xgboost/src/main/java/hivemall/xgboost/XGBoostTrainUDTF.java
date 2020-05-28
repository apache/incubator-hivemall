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
import hivemall.annotations.VisibleForTesting;
import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.hadoop.HadoopUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.OptionUtils;
import hivemall.utils.math.MathUtils;
import hivemall.xgboost.utils.DMatrixBuilder;
import hivemall.xgboost.utils.DenseDMatrixBuilder;
import hivemall.xgboost.utils.NativeLibLoader;
import hivemall.xgboost.utils.SparseDMatrixBuilder;
import hivemall.xgboost.utils.XGBoostUtils;
import matrix4j.utils.lang.ArrayUtils;
import matrix4j.utils.lang.Primitives;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

/**
 * UDTF for train_xgboost
 */
//@formatter:off
@Description(name = "train_xgboost",
        value = "_FUNC_(array<string|double> features, <int|double> target, const string options)"
                + " - Returns a relation consists of <string model_id, array<string> pred_model>",
        extended = "SELECT \n" + 
                "  train_xgboost(features, label, '-objective binary:logistic -iters 10') \n" + 
                "    as (model_id, model)\n" + 
                "from (\n" + 
                "  select features, label\n" + 
                "  from xgb_input\n" + 
                "  cluster by rand(43) -- shuffle\n" + 
                ") shuffled;")
//@formatter:on
public class XGBoostTrainUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(XGBoostTrainUDTF.class);

    // Settings for the XGBoost native library
    static {
        NativeLibLoader.initXGBoost();
    }

    // For input parameters
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private PrimitiveObjectInspector targetOI;

    // For training input buffering
    private boolean denseInput;
    private DMatrixBuilder matrixBuilder;
    private FloatArrayList labels;

    // For XGBoost options
    @Nonnull
    protected final Map<String, Object> params = new HashMap<String, Object>();

    protected int numClass;
    protected ObjectiveType objectiveType = null;

    public enum ObjectiveType {
        regression, binary, multiclass, rank, other;

        @Nonnull
        public static ObjectiveType resolve(@Nonnull String objective) {
            if (objective.startsWith("reg:")) {
                return regression;
            } else if (objective.startsWith("binary:")) {
                return binary;
            } else if (objective.startsWith("multi:")) {
                return multiclass;
            } else if (objective.startsWith("rank:")) {
                return rank;
            } else {
                return other;
            }
        }
    }


    public XGBoostTrainUDTF() {}

    @Override
    protected Options getOptions() {
        final Options opts = new Options();

        opts.addOption("num_round", "iters", true, "Number of boosting iterations [default: 10]");
        opts.addOption("maximize_evaluation_metrics", true,
            "Maximize evaluation metrics [default: false]");
        opts.addOption("num_early_stopping_rounds", true,
            "Minimum rounds required for early stopping [default: 0]");
        opts.addOption("validation_ratio", true,
            "Validation ratio in range [0.0,1.0] [default: 0.2]");

        /** General parameters */
        opts.addOption("booster", true,
            "Set a booster to use, gbtree or gblinear or dart. [default: gbree]");
        opts.addOption("silent", true, "Deprecated. Please use verbosity instead. "
                + "0 means printing running messages, 1 means silent mode [default: 1]");
        opts.addOption("verbosity", true, "Verbosity of printing messages. "
                + "Choices: 0 (silent), 1 (warning), 2 (info), 3 (debug). [default: 0]");
        opts.addOption("disable_default_eval_metric", true,
            "NFlag to disable default metric. Set to >0 to disable. [default: 0]");
        opts.addOption("num_pbuffer", true,
            "Size of prediction buffer [default: set automatically by xgboost]");
        opts.addOption("num_feature", true,
            "Feature dimension used in boosting [default: set automatically by xgboost]");

        /** Parameters among Boosters */
        opts.addOption("lambda", "reg_lambda", true,
            "L2 regularization term on weights. Increasing this value will make model more conservative."
                    + " [default: 1.0 for gbtree, 0.0 for gblinear]");
        opts.addOption("alpha", "reg_alpha", true,
            "L1 regularization term on weights. Increasing this value will make model more conservative."
                    + " [default: 0.0]");
        opts.addOption("updater", true,
            "A comma-separated string that defines the sequence of tree updaters to run. "
                    + "For a full list of valid inputs, please refer to XGBoost Parameters."
                    + " [default: 'grow_colmaker,prune' for gbtree, 'shotgun' for gblinear]");

        /** Parameters for Tree Booster */
        opts.addOption("eta", "learning_rate", true,
            "Step size shrinkage used in update to prevents overfitting [default: 0.3]");
        opts.addOption("gamma", "min_split_loss", true,
            "Minimum loss reduction required to make a further partition on a leaf node of the tree."
                    + " [default: 0.0]");
        opts.addOption("max_depth", true, "Max depth of decision tree [default: 6]");
        opts.addOption("min_child_weight", true,
            "Minimum sum of instance weight (hessian) needed in a child [default: 1.0]");
        opts.addOption("max_delta_step", true,
            "Maximum delta step we allow each tree's weight estimation to be [default: 0]");
        opts.addOption("subsample", true,
            "Subsample ratio of the training instance in range (0.0,1.0] [default: 1.0]");
        opts.addOption("colsample_bytree", true,
            "Subsample ratio of columns when constructing each tree [default: 1.0]");
        opts.addOption("colsample_bylevel", true,
            "Subsample ratio of columns for each level [default: 1.0]");
        opts.addOption("colsample_bynode", true,
            "Subsample ratio of columns for each node [default: 1.0]");
        // tree_method
        opts.addOption("tree_method", true,
            "The tree construction algorithm used in XGBoost. [default: auto, Choices: auto, exact, approx, hist]");
        opts.addOption("sketch_eps", true,
            "This roughly translates into O(1 / sketch_eps) number of bins. \n"
                    + "Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy.\n"
                    + "Only used for tree_method=approx. Usually user does not have to tune this.  [default: 0.03]");
        opts.addOption("scale_pos_weight", true,
            "ontrol the balance of positive and negative weights, useful for unbalanced classes. "
                    + "A typical value to consider: sum(negative instances) / sum(positive instances)"
                    + " [default: 1.0]");
        opts.addOption("refresh_leaf", true,
            "This is a parameter of the refresh updater plugin. "
                    + "When this flag is 1, tree leafs as well as tree nodes’ stats are updated. "
                    + "When it is 0, only node stats are updated. [default: 1]");
        opts.addOption("process_type", true,
            "A type of boosting process to run. [Choices: default, update]");
        opts.addOption("grow_policy", true,
            "Controls a way new nodes are added to the tree. Currently supported only if tree_method is set to hist."
                    + " [default: depthwise, Choices: depthwise, lossguide]");
        opts.addOption("max_leaves", true,
            "Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set. [default: 0]");
        opts.addOption("max_bin", true,
            "Maximum number of discrete bins to bucket continuous features. Only used if tree_method is set to hist."
                    + " [default: 256]");
        opts.addOption("num_parallel_tree", true,
            "Number of parallel trees constructed during each iteration. This option is used to support boosted random forest. "
                    + "Usually no need to tune (default 1 is enough) for gradient boosting trees."
                    + " [default: 1]");

        /** Parameters for Dart Booster (booster=dart) */
        opts.addOption("sample_type", true,
            "Type of sampling algorithm. [Choices: uniform (default), weighted]");
        opts.addOption("normalize_type", true,
            "Type of normalization algorithm. [Choices: tree (default), forest]");
        opts.addOption("rate_drop", true, "Dropout rate in range [0.0, 1.0]. [default: 0.0]");
        opts.addOption("one_drop", true,
            "When this flag is enabled, at least one tree is always dropped during the dropout. "
                    + "0 or 1. [default: 0]");
        opts.addOption("skip_drop", true,
            "Probability of skipping the dropout procedure during a boosting iteration "
                    + "in range [0.0, 1.0]. [default: 0.0]");

        /** Parameters for Linear Booster (booster=gblinear) */
        opts.addOption("lambda_bias", true, "L2 regularization term on bias [default: 0.0]");
        opts.addOption("feature_selector", true, "Feature selection and ordering method. "
                + "[Choices: cyclic (default), shuffle, random, greedy, thrifty]");
        opts.addOption("top_k", true,
            "The number of top features to select in greedy and thrifty feature selector. "
                    + "The value of 0 means using all the features. [default: 0]");

        /** Parameters for Tweedie Regression (objective=reg:tweedie) */
        opts.addOption("tweedie_variance_power", true,
            "Parameter that controls the variance of the Tweedie distribution in range [1.0, 2.0]."
                    + " [default: 1.5]");

        /** Learning Task Parameters */
        opts.addOption("objective", true,
            "Specifies the learning task and the corresponding learning objective. "
                    + "Examples: reg:linear, reg:logistic, multi:softmax. "
                    + "For a full list of valid inputs, refer to XGBoost Parameters. "
                    + "[default: reg:linear]");
        opts.addOption("base_score", true,
            "Initial prediction score of all instances, global bias [default: 0.5]");
        opts.addOption("eval_metric", true,
            "Evaluation metrics for validation data. A default metric is assigned according to the objective:\n"
                    + "- rmse: for regression\n" + "- error: for classification\n"
                    + "- map: for ranking\n"
                    + "For a list of valid inputs, see XGBoost Parameters.");
        opts.addOption("seed", true, "Random number seed. [default: 43]");
        opts.addOption("num_class", true, "Number of classes to classify");

        return opts;
    }

    @Nonnull
    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        final CommandLine cl;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs, 2);
            cl = parseOptions(rawArgs);
        } else {
            cl = parseOptions(""); // use default options
        }

        String objective = cl.getOptionValue("objective");
        if (objective == null) {
            showHelp("Please provide \"-objective XXX\" option in the 3rd argument.\n\n"
                    + "Here is the list of supported objectives: \n"
                    + " - Regression:\n {reg:squarederror, reg:logistic, reg:gamma, reg:tweedie}\n"
                    + " - Binary classification: {binary:logistic, binary:logitraw, binary:hinge}\n"
                    + " - Multiclass classification:\n {multi:softmax, multi:softprob}\n"
                    + " - Ranking:\n {rank:pairwise, rank:ndcg, rank:map}\n"
                    + " - Other:\n {count:poisson, survival:cox}");
        }
        if (objective.equals("reg:squarederror")) {
            // reg:linear is deprecated synonym of reg:squarederror
            // however, reg:squarederror is not supported in xgboost-predictor yet
            // https://github.com/dmlc/xgboost/pull/4267
            objective = "reg:linear";
        }
        final String booster = cl.getOptionValue("booster", "gbtree");

        int numRound = Primitives.parseInt(cl.getOptionValue("num_round"), 10);
        params.put("num_round", numRound);
        params.put("maximize_evaluation_metrics",
            Primitives.parseBoolean(cl.getOptionValue("maximize_evaluation_metrics"), false));
        params.put("num_early_stopping_rounds",
            Primitives.parseInt(cl.getOptionValue("num_early_stopping_rounds"), 0));
        double validationRatio =
                Primitives.parseDouble(cl.getOptionValue("validation_ratio"), 0.2d);
        if (validationRatio < 0.d || validationRatio >= 1.d) {
            throw new UDFArgumentException("Invalid validation_ratio=" + validationRatio);
        }
        params.put("validation_ratio", validationRatio);

        /** General parameters */
        params.put("booster", booster);
        params.put("silent", Primitives.parseInt(cl.getOptionValue("silent"), 1));
        params.put("verbosity", Primitives.parseInt(cl.getOptionValue("verbosity"), 0));
        params.put("nthread", Primitives.parseInt(cl.getOptionValue("nthread"), 1));
        params.put("disable_default_eval_metric",
            Primitives.parseInt(cl.getOptionValue("disable_default_eval_metric"), 0));
        if (cl.hasOption("num_pbuffer")) {
            params.put("num_pbuffer", Integer.valueOf(cl.getOptionValue("num_pbuffer")));
        }
        if (cl.hasOption("num_feature")) {
            params.put("num_feature", Integer.valueOf(cl.getOptionValue("num_feature")));
        }

        /** Parameters for Tree Booster (booster=gbtree) */
        if (booster.equals("gbtree")) {
            params.put("eta", Primitives.parseDouble(cl.getOptionValue("eta"), 0.3d));
            params.put("gamma", Primitives.parseDouble(cl.getOptionValue("gamma"), 0.d));
            params.put("max_depth", Primitives.parseInt(cl.getOptionValue("max_depth"), 6));
            params.put("min_child_weight",
                Primitives.parseDouble(cl.getOptionValue("min_child_weight"), 1.d));
            params.put("max_delta_step",
                Primitives.parseDouble(cl.getOptionValue("max_delta_step"), 0.d));
            params.put("subsample", Primitives.parseDouble(cl.getOptionValue("subsample"), 1.d));
            params.put("colsamle_bytree",
                Primitives.parseDouble(cl.getOptionValue("colsample_bytree"), 1.d));
            params.put("colsamle_bylevel",
                Primitives.parseDouble(cl.getOptionValue("colsamle_bylevel"), 1.d));
            params.put("colsamle_bynode",
                Primitives.parseDouble(cl.getOptionValue("colsamle_bynode"), 1.d));
            params.put("lambda", Primitives.parseDouble(cl.getOptionValue("lambda"), 1.d));
            params.put("alpha", Primitives.parseDouble(cl.getOptionValue("alpha"), 0.d));
            params.put("tree_method", cl.getOptionValue("tree_method", "auto"));
            params.put("sketch_eps",
                Primitives.parseDouble(cl.getOptionValue("sketch_eps"), 0.03d));
            params.put("scale_pos_weight",
                Primitives.parseDouble(cl.getOptionValue("scale_pos_weight"), 1.d));
            params.put("updater", cl.getOptionValue("updater", "grow_colmaker,prune"));
            params.put("refresh_leaf", Primitives.parseInt(cl.getOptionValue("refresh_leaf"), 1));
            params.put("process_type", cl.getOptionValue("process_type", "default"));
            params.put("grow_policy", cl.getOptionValue("grow_policy", "depthwise"));
            params.put("max_leaves", Primitives.parseInt(cl.getOptionValue("max_leaves"), 0));
            params.put("max_bin", Primitives.parseInt(cl.getOptionValue("max_bin"), 256));
            params.put("num_parallel_tree",
                Primitives.parseInt(cl.getOptionValue("num_parallel_tree"), 1));
        }

        /** Parameters for Dart Booster (booster=dart) */
        if (booster.equals("dart")) {
            params.put("sample_type", cl.getOptionValue("sample_type", "uniform"));
            params.put("normalize_type", cl.getOptionValue("normalize_type", "tree"));
            params.put("rate_drop", Primitives.parseDouble(cl.getOptionValue("rate_drop"), 0.d));
            params.put("one_drop", Primitives.parseInt(cl.getOptionValue("one_drop"), 0));
            params.put("skip_drop", Primitives.parseDouble(cl.getOptionValue("skip_drop"), 0.d));
        }

        /** Parameters for Linear Booster (booster=gblinear) */
        if (booster.equals("gblinear")) {
            params.put("lambda", Primitives.parseDouble(cl.getOptionValue("lambda"), 0.d));
            params.put("lambda_bias",
                Primitives.parseDouble(cl.getOptionValue("lambda_bias"), 0.d));
            params.put("alpha", Primitives.parseDouble(cl.getOptionValue("alpha"), 0.d));
            params.put("updater", cl.getOptionValue("updater", "shotgun"));
            params.put("feature_selector", cl.getOptionValue("feature_selector", "cyclic"));
            params.put("top_k", Primitives.parseInt(cl.getOptionValue("top_k"), 0));
        }

        /** Parameters for Tweedie Regression (objective=reg:tweedie) */
        if (objective.equals("reg:tweedie")) {
            params.put("tweedie_variance_power",
                Primitives.parseDouble(cl.getOptionValue("tweedie_variance_power"), 1.5d));
        }

        /** Parameters for Poisson Regression (objective=count:poisson) */
        if (objective.equals("count:poisson")) {
            // max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
            params.put("max_delta_step",
                Primitives.parseDouble(cl.getOptionValue("max_delta_step"), 0.7d));
        }

        /** Learning Task Parameters */
        params.put("objective", objective);
        params.put("base_score", Primitives.parseDouble(cl.getOptionValue("base_score"), 0.5d));
        if (cl.hasOption("eval_metric")) {
            params.put("eval_metric", cl.getOptionValue("eval_metric"));
        }
        params.put("seed", Primitives.parseLong(cl.getOptionValue("seed"), 43L));

        if (cl.hasOption("num_class")) {
            this.numClass = Integer.parseInt(cl.getOptionValue("num_class"));
            params.put("num_class", numClass);
        } else {
            if (objective.startsWith("multi:")) {
                throw new UDFArgumentException(
                    "-num_class is required for multiclass classification");
            }
        }

        if (logger.isInfoEnabled()) {
            logger.info("XGboost training hyperparameters: " + params.toString());
        }

        this.objectiveType = ObjectiveType.resolve(objective);

        return cl;
    }

    @Override
    public StructObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            showHelp("Invalid argment length=" + argOIs.length);
        }
        processOptions(argOIs);

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs, 0);
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        this.featureListOI = listOI;
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.denseInput = true;
            this.matrixBuilder = new DenseDMatrixBuilder(8192);
        } else if (HiveUtils.isStringOI(elemOI)) {
            this.featureElemOI = HiveUtils.asStringOI(elemOI);
            this.denseInput = false;
            this.matrixBuilder = new SparseDMatrixBuilder(8192);
        } else {
            throw new UDFArgumentException(
                "train_xgboost takes array<double> or array<string> for the first argument: "
                        + listOI.getTypeName());
        }
        this.targetOI = HiveUtils.asDoubleCompatibleOI(argOIs, 1);
        this.labels = new FloatArrayList(1024);

        final List<String> fieldNames = new ArrayList<>(2);
        final List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("model");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    /** To validate target range, overrides this method */
    protected float processTargetValue(final float target) throws HiveException {
        switch (objectiveType) {
            case binary: {
                if (target != -1 && target != 0 && target != 1) {
                    throw new UDFArgumentException(
                        "Invalid label value for classification: " + target);
                }
                return target > 0.f ? 1.f : 0.f;
            }
            case multiclass: {
                final int clazz = (int) target;
                if (clazz != target) {
                    throw new UDFArgumentException(
                        "Invalid target value for class label: " + target);
                }
                if (clazz < 0 || clazz >= numClass) {
                    throw new UDFArgumentException("target must be {0.0, ..., "
                            + String.format("%.1f", (numClass - 1.0)) + "}: " + target);
                }
                return target;
            }
            default:
                return target;
        }
    }

    @Override
    public void process(@Nonnull Object[] args) throws HiveException {
        if (args[0] == null) {
            throw new HiveException("array<double> features was null");
        }
        parseFeatures(args[0], matrixBuilder);

        float target = PrimitiveObjectInspectorUtils.getFloat(args[1], targetOI);
        labels.add(processTargetValue(target));
    }

    private void parseFeatures(@Nonnull final Object argObj,
            @Nonnull final DMatrixBuilder builder) {
        if (denseInput) {
            final int length = featureListOI.getListLength(argObj);
            for (int i = 0; i < length; i++) {
                Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    continue;
                }
                float v = PrimitiveObjectInspectorUtils.getFloat(o, featureElemOI);
                builder.nextColumn(i, v);
            }
        } else {
            final int length = featureListOI.getListLength(argObj);
            for (int i = 0; i < length; i++) {
                Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    continue;
                }
                String fv = o.toString();
                builder.nextColumn(fv);
            }
        }
        builder.nextRow();
    }

    @Override
    public void close() throws HiveException {
        final Reporter reporter = getReporter();

        DMatrix dmatrix = null;
        Booster booster = null;
        try {
            dmatrix = matrixBuilder.buildMatrix(labels.toArray(true));
            this.matrixBuilder = null;
            this.labels = null;

            final int round = OptionUtils.getInt(params, "num_round");
            final int earlyStoppingRounds = OptionUtils.getInt(params, "num_early_stopping_rounds");
            if (earlyStoppingRounds > 0) {
                double validationRatio = OptionUtils.getDouble(params, "validation_ratio");
                long seed = OptionUtils.getLong(params, "seed");

                int numRows = (int) dmatrix.rowNum();
                int[] rows = MathUtils.permutation(numRows);
                ArrayUtils.shuffle(rows, new Random(seed));

                int numTest = (int) (numRows * validationRatio);
                DMatrix dtrain = null, dtest = null;
                try {
                    dtest = dmatrix.slice(Arrays.copyOf(rows, numTest));
                    dtrain = dmatrix.slice(Arrays.copyOfRange(rows, numTest, rows.length));
                    booster = train(dtrain, dtest, round, earlyStoppingRounds, params, reporter);
                } finally {
                    XGBoostUtils.close(dtrain);
                    XGBoostUtils.close(dtest);
                }
            } else {
                booster = train(dmatrix, round, params, reporter);
            }
            onFinishTraining(booster);

            // Output the built model
            String modelId = generateUniqueModelId();
            Text predModel = XGBoostUtils.serializeBooster(booster);

            logger.info("model_id:" + modelId.toString() + ", size:" + predModel.getLength());
            forward(new Object[] {modelId, predModel});
        } catch (Throwable e) {
            throw new HiveException(e);
        } finally {
            XGBoostUtils.close(dmatrix);
            XGBoostUtils.close(booster);
        }
    }

    @VisibleForTesting
    protected void onFinishTraining(@Nonnull Booster booster) {}

    @Nonnull
    private static Booster train(@Nonnull final DMatrix dtrain, @Nonnegative final int round,
            @Nonnull final Map<String, Object> params, @Nullable final Reporter reporter)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException,
            InstantiationException, XGBoostError {
        final Counters.Counter iterCounter = (reporter == null) ? null
                : reporter.getCounter("hivemall.XGBoostTrainUDTF$Counter", "iteration");

        final Booster booster = XGBoostUtils.createBooster(dtrain, params);
        for (int iter = 0; iter < round; iter++) {
            reportProgress(reporter);
            setCounterValue(iterCounter, iter + 1);

            booster.update(dtrain, iter);
        }
        return booster;
    }

    @Nonnull
    private static Booster train(@Nonnull final DMatrix dtrain, @Nonnull final DMatrix dtest,
            @Nonnegative final int round, @Nonnegative final int earlyStoppingRounds,
            @Nonnull final Map<String, Object> params, @Nullable final Reporter reporter)
            throws NoSuchMethodException, IllegalAccessException, InvocationTargetException,
            InstantiationException, XGBoostError {
        final Counters.Counter iterCounter = (reporter == null) ? null
                : reporter.getCounter("hivemall.XGBoostTrainUDTF$Counter", "iteration");

        final Booster booster = XGBoostUtils.createBooster(dtrain, params);

        final boolean maximizeEvaluationMetrics =
                OptionUtils.getBoolean(params, "maximize_evaluation_metrics");
        float bestScore = maximizeEvaluationMetrics ? -Float.MAX_VALUE : Float.MAX_VALUE;
        int bestIteration = 0;

        final float[] metricsOut = new float[1];
        for (int iter = 0; iter < round; iter++) {
            reportProgress(reporter);
            setCounterValue(iterCounter, iter + 1);

            booster.update(dtrain, iter);

            String evalInfo =
                    booster.evalSet(new DMatrix[] {dtest}, new String[] {"test"}, iter, metricsOut);
            logger.info(evalInfo);

            final float score = metricsOut[0];
            if (maximizeEvaluationMetrics) {
                // Update best score if the current score is better (no update when equal)
                if (score > bestScore) {
                    bestScore = score;
                    bestIteration = iter;
                }
            } else {
                if (score < bestScore) {
                    bestScore = score;
                    bestIteration = iter;
                }
            }

            if (shouldEarlyStop(earlyStoppingRounds, iter, bestIteration)) {
                logger.info(
                    String.format("early stopping after %d rounds away from the best iteration",
                        earlyStoppingRounds));
                break;
            }
        }

        return booster;
    }

    private static boolean shouldEarlyStop(final int earlyStoppingRounds, final int iter,
            final int bestIteration) {
        return iter - bestIteration >= earlyStoppingRounds;
    }

    @Nonnull
    private static String generateUniqueModelId() {
        return "xgbmodel-" + HadoopUtils.getUniqueTaskIdString();
    }

}
