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

import hivemall.mix.MixMessage.MixEventName;
import hivemall.mix.client.MixClient;
import hivemall.model.DenseModel;
import hivemall.model.NewDenseModel;
import hivemall.model.NewSpaceEfficientDenseModel;
import hivemall.model.NewSparseModel;
import hivemall.model.PredictionModel;
import hivemall.model.SpaceEfficientDenseModel;
import hivemall.model.SparseModel;
import hivemall.model.SynchronizedModelWrapper;
import hivemall.optimizer.DenseOptimizerFactory;
import hivemall.optimizer.Optimizer;
import hivemall.optimizer.SparseOptimizerFactory;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;

import java.util.Map;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

public abstract class LearnerBaseUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(LearnerBaseUDTF.class);

    protected final boolean enableNewModel;
    protected boolean dense_model;
    protected int model_dims;
    protected boolean disable_halffloat;
    protected boolean is_mini_batch;
    protected int mini_batch_size;
    protected String mixConnectInfo;
    protected String mixSessionName;
    protected int mixThreshold;
    protected boolean mixCancel;
    protected boolean ssl;

    @Nullable
    protected MixClient mixClient;

    public LearnerBaseUDTF(boolean enableNewModel) {
        this.enableNewModel = enableNewModel;
    }

    protected boolean useCovariance() {
        return false;
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("dense", "densemodel", false, "Use dense model or not");
        opts.addOption("dims", "feature_dimensions", true,
            "The dimension of model [default: 16777216 (2^24)]");
        opts.addOption("disable_halffloat", false,
            "Toggle this option to disable the use of SpaceEfficientDenseModel");
        opts.addOption("mini_batch", "mini_batch_size", true,
            "Mini batch size [default: 1]. Expecting the value in range [1,100] or so.");
        opts.addOption("mix", "mix_servers", true, "Comma separated list of MIX servers");
        opts.addOption("mix_session", "mix_session_name", true,
            "Mix session name [default: ${mapred.job.id}]");
        opts.addOption("mix_threshold", true,
            "Threshold to mix local updates in range (0,127] [default: 3]");
        opts.addOption("mix_cancel", "enable_mix_canceling", false, "Enable mix cancel requests");
        opts.addOption("ssl", false, "Use SSL for the communication with mix servers");
        return opts;
    }

    @Nullable
    @Override
    protected CommandLine processOptions(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        boolean denseModel = false;
        int modelDims = -1;
        boolean disableHalfFloat = false;
        int miniBatchSize = 1;
        String mixConnectInfo = null;
        String mixSessionName = null;
        int mixThreshold = -1;
        boolean mixCancel = false;
        boolean ssl = false;

        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);

            denseModel = cl.hasOption("dense");
            if (denseModel) {
                modelDims = Primitives.parseInt(cl.getOptionValue("dims"), 16777216);
            }
            disableHalfFloat = cl.hasOption("disable_halffloat");

            miniBatchSize = Primitives.parseInt(cl.getOptionValue("mini_batch_size"), miniBatchSize);
            if (miniBatchSize <= 0) {
                throw new UDFArgumentException("mini_batch_size must be greater than 0: "
                        + miniBatchSize);
            }

            mixConnectInfo = cl.getOptionValue("mix");
            mixSessionName = cl.getOptionValue("mix_session");
            mixThreshold = Primitives.parseInt(cl.getOptionValue("mix_threshold"), 3);
            if (mixThreshold > Byte.MAX_VALUE) {
                throw new UDFArgumentException("mix_threshold must be in range (0,127]: "
                        + mixThreshold);
            }
            mixCancel = cl.hasOption("mix_cancel");
            ssl = cl.hasOption("ssl");
        }

        this.dense_model = denseModel;
        this.model_dims = modelDims;
        this.disable_halffloat = disableHalfFloat;
        this.is_mini_batch = miniBatchSize > 1;
        this.mini_batch_size = miniBatchSize;
        this.mixConnectInfo = mixConnectInfo;
        this.mixSessionName = mixSessionName;
        this.mixThreshold = mixThreshold;
        this.mixCancel = mixCancel;
        this.ssl = ssl;
        return cl;
    }

    @Nullable
    protected PredictionModel createModel() {
        if (enableNewModel) {
            return createNewModel(null);
        } else {
            return createOldModel(null);
        }
    }

    @Nonnull
    private final PredictionModel createOldModel(@Nullable String label) {
        PredictionModel model;
        final boolean useCovar = useCovariance();
        if (dense_model) {
            if (disable_halffloat == false && model_dims > 16777216) {
                logger.info("Build a space efficient dense model with " + model_dims
                        + " initial dimensions" + (useCovar ? " w/ covariances" : ""));
                model = new SpaceEfficientDenseModel(model_dims, useCovar);
            } else {
                logger.info("Build a dense model with initial with " + model_dims
                        + " initial dimensions" + (useCovar ? " w/ covariances" : ""));
                model = new DenseModel(model_dims, useCovar);
            }
        } else {
            int initModelSize = getInitialModelSize();
            logger.info("Build a sparse model with initial with " + initModelSize
                    + " initial dimensions");
            model = new SparseModel(initModelSize, useCovar);
        }
        if (mixConnectInfo != null) {
            model.configureClock();
            model = new SynchronizedModelWrapper(model);
            MixClient client = configureMixClient(mixConnectInfo, label, model);
            model.configureMix(client, mixCancel);
            this.mixClient = client;
        }
        assert (model != null);
        return model;
    }

    @Nonnull
    private final PredictionModel createNewModel(@Nullable String label) {
        PredictionModel model;
        final boolean useCovar = useCovariance();
        if (dense_model) {
            if (disable_halffloat == false && model_dims > 16777216) {
                logger.info("Build a space efficient dense model with " + model_dims
                        + " initial dimensions" + (useCovar ? " w/ covariances" : ""));
                model = new NewSpaceEfficientDenseModel(model_dims, useCovar);
            } else {
                logger.info("Build a dense model with initial with " + model_dims
                        + " initial dimensions" + (useCovar ? " w/ covariances" : ""));
                model = new NewDenseModel(model_dims, useCovar);
            }
        } else {
            int initModelSize = getInitialModelSize();
            logger.info("Build a sparse model with initial with " + initModelSize
                    + " initial dimensions");
            model = new NewSparseModel(initModelSize, useCovar);
        }
        if (mixConnectInfo != null) {
            model.configureClock();
            model = new SynchronizedModelWrapper(model);
            MixClient client = configureMixClient(mixConnectInfo, label, model);
            model.configureMix(client, mixCancel);
            this.mixClient = client;
        }
        assert (model != null);
        return model;
    }

    @Nonnull
    protected final Optimizer createOptimizer(@CheckForNull Map<String, String> options) {
        Preconditions.checkNotNull(options);
        if (dense_model) {
            return DenseOptimizerFactory.create(model_dims, options);
        } else {
            return SparseOptimizerFactory.create(model_dims, options);
        }
    }

    @Nonnull
    protected MixClient configureMixClient(@Nonnull String connectURIs, @Nullable String label,
            @Nonnull PredictionModel model) {
        String jobId = (mixSessionName == null) ? MixClient.DUMMY_JOB_ID : mixSessionName;
        if (label != null) {
            jobId = jobId + '-' + label;
        }
        MixEventName event = useCovariance() ? MixEventName.argminKLD : MixEventName.average;
        MixClient client = new MixClient(event, jobId, connectURIs, ssl, mixThreshold, model);
        logger.info("Successfully configured mix client: " + connectURIs);
        return client;
    }

    protected int getInitialModelSize() {
        return 16384;
    }

    @Nonnull
    protected ObjectInspector getFeatureOutputOI(@Nonnull PrimitiveObjectInspector featureInputOI)
            throws UDFArgumentException {
        if (dense_model) {
            // TODO validation
            return PrimitiveObjectInspectorFactory.javaIntObjectInspector; // see DenseModel
        }
        return ObjectInspectorUtils.getStandardObjectInspector(featureInputOI);
    }

    @Override
    public void close() throws HiveException {
        if (mixClient != null) {
            IOUtils.closeQuietly(mixClient);
            this.mixClient = null;
        }
    }

}
