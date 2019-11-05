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
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
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

    // For input buffer
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

        /** General parameters */
        opts.addOption("booster", true,
            "Set a booster to use, gbtree or gblinear. [default: gbree]");
        opts.addOption("num_round", true, "Number of boosting iterations [default: 8]");
        opts.addOption("silent", true,
            "0 means printing running messages, 1 means silent mode [default: 1]");
        opts.addOption("nthread", true,
            "Number of parallel threads used to run xgboost [default: 1]");
        opts.addOption("num_pbuffer", true,
            "Size of prediction buffer [set automatically by xgboost]");
        opts.addOption("num_feature", true,
            "Feature dimension used in boosting [default: set automatically by xgboost]");

        /** Parameters for both boosters */
        opts.addOption("alpha", true, "L1 regularization term on weights [default: 0.0]");
        opts.addOption("lambda", true,
            "L2 regularization term on weights [default: 1.0 for gbtree, 0.0 for gblinear]");

        /** Parameters for Tree Booster */
        opts.addOption("eta", true,
            "Step size shrinkage used in update to prevents overfitting [default: 0.3]");
        opts.addOption("gamma", true,
            "Minimum loss reduction required to make a further partition on a leaf node of the tree [default: 0.0]");
        opts.addOption("max_depth", true, "Max depth of decision tree [default: 6]");
        opts.addOption("min_child_weight", true,
            "Minimum sum of instance weight(hessian) needed in a child [default: 1]");
        opts.addOption("max_delta_step", true,
            "Maximum delta step we allow each tree's weight estimation to be [default: 0]");
        opts.addOption("subsample", true,
            "Subsample ratio of the training instance [default: 1.0]");
        opts.addOption("colsample_bytree", true,
            "Subsample ratio of columns when constructing each tree [default: 1.0]");
        opts.addOption("colsample_bylevel", true,
            "Subsample ratio of columns for each split, in each level [default: 1.0]");

        /** Parameters for Linear Booster */
        opts.addOption("lambda_bias", true, "L2 regularization term on bias [default: 0.0]");

        /** Learning Task Parameters */
        opts.addOption("base_score", true,
            "Initial prediction score of all instances, global bias [default: 0.5]");
        opts.addOption("eval_metric", true,
            "Evaluation metrics for validation data [default according to objective]");

        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        if (argOIs.length >= 3) {
            final String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = this.parseOptions(rawArgs);

            /** General parameters */
            if (cl.hasOption("booster")) {
                params.put("booster", cl.getOptionValue("booster"));
            }
            if (cl.hasOption("num_round")) {
                params.put("num_round", Integer.valueOf(cl.getOptionValue("num_round")));
            }
            if (cl.hasOption("silent")) {
                params.put("silent", Integer.valueOf(cl.getOptionValue("silent")));
            }
            if (cl.hasOption("nthread")) {
                params.put("nthread", Integer.valueOf(cl.getOptionValue("nthread")));
            }
            if (cl.hasOption("num_pbuffer")) {
                params.put("num_pbuffer", Integer.valueOf(cl.getOptionValue("num_pbuffer")));
            }
            if (cl.hasOption("num_feature")) {
                params.put("num_feature", Integer.valueOf(cl.getOptionValue("num_feature")));
            }

            /** Parameters for both boosters */
            if (cl.hasOption("alpha")) {
                params.put("alpha", Double.valueOf(cl.getOptionValue("alpha")));
            }
            if (cl.hasOption("lambda")) {
                params.put("lambda", Double.valueOf(cl.getOptionValue("lambda")));
            }

            /** Parameters for Tree Booster */
            if (cl.hasOption("eta")) {
                params.put("eta", Double.valueOf(cl.getOptionValue("eta")));
            }
            if (cl.hasOption("gamma")) {
                params.put("gamma", Double.valueOf(cl.getOptionValue("gamma")));
            }
            if (cl.hasOption("max_depth")) {
                params.put("max_depth", Integer.valueOf(cl.getOptionValue("max_depth")));
            }
            if (cl.hasOption("min_child_weight")) {
                params.put("min_child_weight",
                    Integer.valueOf(cl.getOptionValue("min_child_weight")));
            }
            if (cl.hasOption("max_delta_step")) {
                params.put("max_delta_step", Integer.valueOf(cl.getOptionValue("max_delta_step")));
            }
            if (cl.hasOption("subsample")) {
                params.put("subsample", Double.valueOf(cl.getOptionValue("subsample")));
            }
            if (cl.hasOption("colsample_bytree")) {
                params.put("colsamle_bytree",
                    Double.valueOf(cl.getOptionValue("colsample_bytree")));
            }
            if (cl.hasOption("colsample_bylevel")) {
                params.put("colsamle_bylevel",
                    Double.valueOf(cl.getOptionValue("colsample_bylevel")));
            }

            /** Parameters for Linear Booster */
            if (cl.hasOption("lambda_bias")) {
                params.put("lambda_bias", Double.valueOf(cl.getOptionValue("lambda_bias")));
            }

            /** Learning Task Parameters */
            if (cl.hasOption("base_score")) {
                params.put("base_score", Double.valueOf(cl.getOptionValue("base_score")));
            }
            if (cl.hasOption("eval_metric")) {
                params.put("eval_metric", cl.getOptionValue("eval_metric"));
            }
        }

        return cl;
    }

    @Override
    public StructObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentLengthException("Invalid argment length=" + argOIs.length);
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
        fieldNames.add("pred_model");
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
            DMatrix dtrain = matrixBuilder.buildMatrix(labels.toArray());
            this.matrixBuilder = null;
            this.labels = null;
            Booster booster = XGBoostUtils.createXGBooster(dtrain, params);

            // Kick off training with XGBoost
            final int round = (Integer) params.get("num_round");
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
