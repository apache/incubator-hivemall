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

@Description(name = "train_plsa", value = "_FUNC_(array<string> words[, const string options])"
        + " - Returns a relation consists of <int topic, string word, float score>")
public final class PLSAUDTF extends ProbabilisticTopicModelBaseUDTF {

    public static final float DEFAULT_ALPHA = 0.5f;
    public static final double DEFAULT_DELTA = 1E-3d;

    // Options
    protected float alpha;
    protected double delta;

    public PLSAUDTF() {
        super();

        this.alpha = DEFAULT_ALPHA;
        this.delta = DEFAULT_DELTA;
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("alpha", true, "The hyperparameter for P(w|z) update [default: 0.5]");
        opts.addOption("delta", true, "Check convergence in the expectation step [default: 1E-3]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        if (cl != null) {
            this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), DEFAULT_ALPHA);
            this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), DEFAULT_DELTA);
        }

        return cl;
    }

    protected AbstractProbabilisticTopicModel createModel() {
        return new IncrementalPLSAModel(topics, alpha, delta);
    }

}
