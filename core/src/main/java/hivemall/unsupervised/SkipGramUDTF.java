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
package hivemall.unsupervised;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.commons.cli.Options;

public class SkipGramUDTF extends Word2vecBaseUDTF {

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("lr", "learningRate", true, "initial learning rate of SGD [default: 0.025]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        float lr = 0.025f;

        if (argOIs.length >= 6) {
            String rawArgs = HiveUtils.getConstString(argOIs[5]);
            cl = parseOptions(rawArgs);

            lr = Primitives.parseFloat(cl.getOptionValue("lr"), lr);
            if (lr < 0.d) {
                throw new UDFArgumentException("Argument `float lr` must be positive: " + lr);
            }
        }

        this.currentLR = this.startingLR = lr;
        return cl;
    }

    protected AbstractWord2vecModel createModel() {
        return new SkipGramModel(dim, win, neg, numTrainWords, S, A);
    }
}
