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
        opts.addOption("optimizer", "opt", true,
            "Optimizer to update weights [default: adagrad, sgd, adadelta, adam]");
        opts.addOption("eps", true, "Denominator value of AdaDelta/AdaGrad [default 1e-6]");
        opts.addOption("rho", "decay", true, "Decay rate of AdaDelta [default 0.95]");
        // regularization
        opts.addOption("regularization", "reg", true, "Regularization type [default: rda, l1, l2]");
        opts.addOption("lambda", true, "Regularization term [default 0.0001]");
        // learning rates
        opts.addOption("eta", true, "Learning rate scheme [default: inverse/inv, fixed, simple]");
        opts.addOption("eta0", true, "The initial learning rate [default 0.1]");
        opts.addOption("t", "total_steps", true, "a total of n_samples * epochs time steps");
        opts.addOption("power_t", true,
            "The exponent for inverse scaling learning rate [default 0.1]");
        // other
        opts.addOption("scale", true, "Scaling factor for cumulative weights [100.0]");
    }

    public static void propcessOptions(@Nullable CommandLine cl,
            @Nonnull Map<String, String> options) {
        if (cl != null) {
            for (String arg : cl.getArgs()) {
                options.put(arg, cl.getOptionValue(arg));
            }
        }
    }

}
