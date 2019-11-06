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
import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.hadoop.HadoopUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.xgboost.utils.DMatrixBuilder;
import hivemall.xgboost.utils.DenseDMatrixBuilder;
import hivemall.xgboost.utils.SparseDMatrixBuilder;
import matrix4j.utils.lang.Primitives;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

/**
 * This is a base class to handle the options for XGBoost and provide common functions among various
 * tasks.
 */
public abstract class XGBoostBaseUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(XGBoostBaseUDTF.class);

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

    // XGBoost options can be found in https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    // Most of default parameters are set along with the official one.
    {
        /** General parameters */
        params.put("booster", "gbtree");
        params.put("num_round", 8);
        params.put("silent", 1);
        // Set to 1 by default because most of distributed systems assume
        // each worker has a single vcore.
        params.put("nthread", 1);

        /** Parameters for both boosters */
        params.put("alpha", 0.0);
        // This default value depends on a booster type
        // params.put("lambda", 0.0);

        /** Parameters for Tree Booster */
        params.put("eta", 0.3);
        params.put("gamma", 0.0);
        params.put("max_depth", 6);
        params.put("min_child_weight", 1);
        params.put("max_delta_step", 0);
        params.put("subsample", 1.0);
        params.put("colsample_bytree", 1.0);
        params.put("colsample_bylevel", 1.0);
        // The memory-based version of XGBoost only supports `exact`
        params.put("tree_method", "exact");

        /** Learning Task Parameters */
        params.put("base_score", 0.5);
    }

    public XGBoostBaseUDTF() {}

    @Override
    protected Options getOptions() {
        final Options opts = new Options();

        opts.addOption("num_round", true, "Number of boosting iterations [default: 10]");

        /** General parameters */
        opts.addOption("booster", true,
            "Set a booster to use, gbtree or gblinear or dart. [default: gbree]");
        opts.addOption("silent", true, "Deprecated. Please use verbosity instead. "
                + "0 means printing running messages, 1 means silent mode [default: 1]");
        opts.addOption("verbosity", true, "Verbosity of printing messages. "
                + "Choices: 0 (silent), 1 (warning), 2 (info), 3 (debug). [default: 0]");
        opts.addOption("nthread", true,
            "Number of parallel threads used to run xgboost [default: 1]");
        opts.addOption("disable_default_eval_metric", true,
            "NFlag to disable default metric. Set to >0 to disable. [default: 0]");
        opts.addOption("num_pbuffer", true,
            "Size of prediction buffer [default: set automatically by xgboost]");
        opts.addOption("num_feature", true,
            "Feature dimension used in boosting [default: set automatically by xgboost]");

        /** Parameters among Boosters */
        opts.addOption("lambda", "reg_lambda", true,
            "L2 regularization term on weights [default: 1.0 for gbtree, 0.0 for gblinear]");
        opts.addOption("alpha", "reg_alpha", true,
            "L1 regularization term on weights [default: 0.0]");
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
        opts.addOption("tree_method", true,
            "The tree construction algorithm used in XGBoost. Choices: auto, exact, approx, hist."
                    + " [default: auto]");
        opts.addOption("sketch_eps", true,
            "Used only for approximate greedy algorithm. This translates into O(1 / sketch_eps) number of bins."
                    + " [default: 0.03]");
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
            "Controls a way new nodes are added to the tree. "
                    + "Currently supported only if tree_method is set to hist."
                    + " [Choices: depthwise (default), lossguide]");
        opts.addOption("max_leaves", true,
            "Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set."
                    + " [default: 0]");
        opts.addOption("max_bin", true,
            "Maximum number of discrete bins to bucket continuous features. "
                    + "Only used if tree_method is set to hist. [default: 256]");
        opts.addOption("num_parallel_tree", true,
            "Number of parallel trees constructed during each iteration. "
                    + "This option is used to support boosted random forest. [default: 1]");

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
                    + "[default: reg:squarederror]");
        opts.addOption("base_score", true,
            "Initial prediction score of all instances, global bias [default: 0.5]");
        opts.addOption("eval_metric", true,
            "Evaluation metrics for validation data. A default metric is assigned according to the objective:\n"
                    + "- rmse: for regression\n" + "- error: for classification\n"
                    + "- map: for ranking\n"
                    + "For a list of valid inputs, see XGBoost Parameters.");
        opts.addOption("seed", true, "Random number seed. [default: 0]");

        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = this.parseOptions(rawArgs);

            final String objective = cl.getOptionValue("objective", "reg:squarederror");
            params.put("num_round", Primitives.parseInt(cl.getOptionValue("num_round"), 10));

            /** General parameters */
            final String booster = cl.getOptionValue("booster", "gbtree");
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
                    Primitives.parseInt(cl.getOptionValue("max_delta_step"), 0));
                params.put("subsample",
                    Primitives.parseDouble(cl.getOptionValue("subsample"), 1.d));
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
                params.put("refresh_leaf",
                    Primitives.parseInt(cl.getOptionValue("refresh_leaf"), 1));
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
                params.put("rate_drop",
                    Primitives.parseDouble(cl.getOptionValue("rate_drop"), 0.d));
                params.put("one_drop", Primitives.parseInt(cl.getOptionValue("one_drop"), 0));
                params.put("skip_drop",
                    Primitives.parseDouble(cl.getOptionValue("skip_drop"), 0.d));
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

            /** Learning Task Parameters */
            params.put("base_score", Primitives.parseDouble(cl.getOptionValue("base_score"), 0.5d));
            if (cl.hasOption("eval_metric")) {
                params.put("eval_metric", cl.getOptionValue("eval_metric"));
            }
            params.put("seed", Primitives.parseInt(cl.getOptionValue("seed"), 0));
        }

        return cl;
    }

    @Override
    public StructObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            showHelp("Invalid argment length=" + argOIs.length);
        }
        processOptions(argOIs);

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[0]);
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
                "_FUNC_ takes double[] or string[] for the first argument: "
                        + listOI.getTypeName());
        }
        this.targetOI = HiveUtils.asDoubleCompatibleOI(argOIs[1]);

        final List<String> fieldNames = new ArrayList<>(2);
        final List<ObjectInspector> fieldOIs = new ArrayList<>(2);
        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("model");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaByteArrayObjectInspector);
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    /** To validate target range, overrides this method */
    protected void checkTargetValue(float target) throws HiveException {}

    @Override
    public void process(@Nonnull Object[] args) throws HiveException {
        if (args[0] == null) {
            throw new HiveException("array<double> features was null");
        }
        parseFeatures(args[0], matrixBuilder);

        float target = PrimitiveObjectInspectorUtils.getFloat(args[1], targetOI);
        checkTargetValue(target);
        labels.add(target);
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
        try {
            final DMatrix dtrain = matrixBuilder.buildMatrix(labels.toArray());
            this.matrixBuilder = null;
            this.labels = null;
            final Booster booster = XGBoostUtils.createXGBooster(dtrain, params);

            // Kick off training with XGBoost
            final int round = ((Integer) params.get("num_round")).intValue();
            for (int i = 0; i < round; i++) {
                booster.update(dtrain, i);
            }

            // Output the built model
            String modelId = generateUniqueModelId();
            byte[] predModel = booster.toByteArray();
            logger.info("model_id:" + modelId.toString() + " size:" + predModel.length);
            forward(new Object[] {modelId, predModel});
        } catch (Throwable e) {
            throw new HiveException(e);
        }
    }

    @Nonnull
    private static String generateUniqueModelId() {
        return "xgbmodel-" + HadoopUtils.getUniqueTaskIdString();
    }

}
