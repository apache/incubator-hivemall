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
package hivemall.optimizer;

import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public final class OptimizerOptions {

    private OptimizerOptions() {}

    @Nonnull
    public static Map<String, String> create() {
        Map<String, String> opts = new HashMap<String, String>();
        opts.put("optimizer", "adagrad");
        opts.put("regularization", "RDA");
        return opts;
    }

    public static void setup(@Nonnull Options opts) {
        opts.addOption("opt", "optimizer", true, "Optimizer to update weights "
                + "[default: adagrad, sgd, momentum, nesterov, rmsprop, rmspropgraves, adadelta, adam, eve, adam_hd]");
        // hyperparameters
        opts.addOption("eps", true,
            "Denominator value of AdaDelta/AdaGrad/Adam [default: 1e-8 (AdaDelta/Adam), 1.0 (Adagrad)]");
        opts.addOption("rho", "decay", true,
            " Exponential decay rate of the first and second order moments [default 0.95 (AdaDelta, rmsprop)]");
        // regularization
        opts.addOption("reg", "regularization", true,
            "Regularization type [default: rda, l1, l2, elasticnet]");
        opts.addOption("l1_ratio", true,
            "Ratio of L1 regularizer as a part of Elastic Net regularization [default: 0.5]");
        opts.addOption("lambda", true, "Regularization term [default 0.0001]");
        // learning rates
        opts.addOption("eta", true, "Learning rate scheme [default: inverse/inv, fixed, simple]");
        opts.addOption("eta0", true,
            "The initial learning rate [default: " + EtaEstimator.DEFAULT_ETA0 + "]");
        opts.addOption("t", "total_steps", true, "a total of n_samples * epochs time steps");
        opts.addOption("power_t", true, "The exponent for inverse scaling learning rate [default: "
                + EtaEstimator.DEFAULT_POWER_T + "]");
        opts.addOption("alpha", true,
            "Coefficient of learning rate [default: 1.0 (adam/RMSPropGraves), 0.02 (AdamHD/Nesterov)]");
        // ADAM hyperparameters
        opts.addOption("beta1", "momentum", true,
            "Exponential decay rate of the first order moment used in Adam [default: 0.9]");
        opts.addOption("beta2", true,
            "Exponential decay rate of the second order moment used in Adam [default: 0.999]");
        opts.addOption("decay", false, "Weight decay rate [default: 0.0]");
        opts.addOption("amsgrad", false, "Whether to use AMSGrad variant of Adam");
        // ADAM-HD hyperparameters
        opts.addOption("beta", true, "Hyperparameter for tuning alpha in Adam-HD [default: 1e-6f]");
        // Eve hyperparameters
        opts.addOption("beta3", true, "Exponential decay rate of alpha value  [default: 0.999]");
        opts.addOption("c", true,
            "Clipping constant of alpha used in Eve optimizer so that clipped [default: 10]");
        // other
        opts.addOption("scale", true, "Scaling factor for cumulative weights [100.0]");
    }

    public static void processOptions(@Nullable CommandLine cl,
            @Nonnull Map<String, String> options) {
        if (cl == null) {
            return;
        }
        for (Option opt : cl.getOptions()) {
            String optName = opt.getLongOpt();
            if (optName == null) {
                optName = opt.getOpt();
            }
            options.put(optName, opt.getValue());
        }
    }

}
