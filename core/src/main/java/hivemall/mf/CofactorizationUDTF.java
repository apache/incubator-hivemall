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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.NioFixedSegment;
import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;

import java.nio.ByteBuffer;

public class CofactorizationUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(CofactorizationUDTF.class);
    private static final int RECORD_BYTES = (Integer.SIZE + Integer.SIZE + Double.SIZE) / 8;

    // Option variables
    /** The number of latent factors */
    protected int factor;
    /** The regularization factor */
    protected float lambda;
    /** The scaling hyperparameter for zero entries in the rank matrix */
    protected float scale_zero;
    /** The scaling hyperparameter for non-zero entries in the rank matrix */
    protected float scale_nonzero;
    /** The preferred size of the batch for training */
    protected int batchSize;
    /** The initial mean rating */
    protected float meanRating;
    /** Whether update (and return) the mean rating or not */
    protected boolean updateMeanRating;
    /** The number of iterations */
    protected int iterations;
    /** Whether to use bias clause */
    protected boolean useBiasClause;

    /** Initialization strategy of rank matrix */
    protected CofactorModel.RankInitScheme rankInit;

    // Model itself
    protected CofactorModel model;

    // Variable managing status of learning
    /** The number of processed training examples */
    protected long count;
    protected ConversionState cvState;

    // Input OIs and Context
    protected PrimitiveObjectInspector userOI;
    protected PrimitiveObjectInspector itemOI;
    protected PrimitiveObjectInspector ratingOI;

    // Used for iterations
    protected NioFixedSegment fileIO;
    protected ByteBuffer inputBuf;
    private long lastWritePos;

    private float[] userProbe, itemProbe;


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
        opts.addOption("b", "batch_size", true, "The batch size for training [default: 1024]");
        opts.addOption("mu", "mean_rating", true, "The mean rating [default: 0.0]");
        opts.addOption("update_mean", "update_mu", false,
                "Whether update (and return) the mean rating or not");
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
            this.lambda = Primitives.parseFloat(cl.getOptionValue("lambda"), 0.03f);
            this.scale_zero = Primitives.parseFloat(cl.getOptionValue("scale_zero"), 0.1f);
            this.scale_nonzero = Primitives.parseFloat(cl.getOptionValue("scale_nonzero"), 1.0f);
            this.batchSize = Primitives.parseInt(cl.getOptionValue("batch_size"), 1024);
            this.meanRating = Primitives.parseFloat(cl.getOptionValue("mu"), 0.f);
            this.updateMeanRating = cl.hasOption("update_mean");
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
            if (noBias && updateMeanRating) {
                throw new UDFArgumentException(
                        "Cannot set both `update_mean` and `no_bias` option");
            }
        }
        this.rankInit = CofactorModel.RankInitScheme.resolve(rankInitOpt);
        rankInit.setMaxInitValue(maxInitValue);
        initStdDev = Math.max(initStdDev, 1.0d / factor);
        rankInit.setInitStdDev(initStdDev);
        this.cvState = new ConversionState(conversionCheck, convergenceRate);
        return cl;
    }

    @Override
    public void process(Object[] objects) throws HiveException {
        
    }

    @Override
    public void close() throws HiveException {

    }
}
