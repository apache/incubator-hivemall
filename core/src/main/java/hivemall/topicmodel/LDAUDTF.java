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
package hivemall.topicmodel;

import hivemall.utils.lang.Primitives;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;

@Description(name = "train_lda", value = "_FUNC_(array<string> words[, const string options])"
        + " - Returns a relation consists of <int topic, string word, float score>")
public final class LDAUDTF extends ProbabilisticTopicModelBaseUDTF {

    public static final double DEFAULT_DELTA = 1E-3d;

    // Options
    protected float alpha;
    protected float eta;
    protected long numDocs;
    protected double tau0;
    protected double kappa;
    protected double delta;

    public LDAUDTF() {
        super();

        this.alpha = 1.f / topics;
        this.eta = 1.f / topics;
        this.numDocs = 0L;
        this.tau0 = 64.d;
        this.kappa = 0.7;
        this.delta = DEFAULT_DELTA;
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("alpha", true, "The hyperparameter for theta [default: 1/k]");
        opts.addOption("eta", true, "The hyperparameter for beta [default: 1/k]");
        opts.addOption("d", "num_docs", true, "The total number of documents [default: auto]");
        opts.addOption("tau", "tau0", true,
            "The parameter which downweights early iterations [default: 64.0]");
        opts.addOption("kappa", true,
            "Exponential decay rate (i.e., learning rate) [default: 0.7]");
        opts.addOption("delta", true, "Check convergence in the expectation step [default: 1E-3]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        if (cl != null) {
            this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), 1.f / topics);
            this.eta = Primitives.parseFloat(cl.getOptionValue("eta"), 1.f / topics);
            this.numDocs = Primitives.parseLong(cl.getOptionValue("num_docs"), 0L);
            this.tau0 = Primitives.parseDouble(cl.getOptionValue("tau0"), 64.d);
            if (tau0 <= 0.d) {
                throw new UDFArgumentException("'-tau0' must be positive: " + tau0);
            }
            this.kappa = Primitives.parseDouble(cl.getOptionValue("kappa"), 0.7d);
            if (kappa <= 0.5 || kappa > 1.d) {
                throw new UDFArgumentException("'-kappa' must be in (0.5, 1.0]: " + kappa);
            }
            this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), DEFAULT_DELTA);
        }

        return cl;
    }

    protected AbstractProbabilisticTopicModel createModel() {
        return new OnlineLDAModel(topics, alpha, eta, numDocs, tau0, kappa, delta);
    }
}
