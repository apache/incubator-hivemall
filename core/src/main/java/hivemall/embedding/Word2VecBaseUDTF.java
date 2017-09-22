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
package hivemall.embedding;

import hivemall.UDTFWithOptions;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import java.util.Map;

public abstract class Word2VecBaseUDTF extends UDTFWithOptions {
    protected transient AbstractWord2VecModel model;

    protected int dim;
    protected Map<String, Integer> word2index;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("dim", "dimension", true, "the number of vector dimension [default: 100]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        int dim = 100;

        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);

            dim = Primitives.parseInt(cl.getOptionValue("dim"), dim);
            if (dim <= 0.d) {
                throw new UDFArgumentException("Argument `int dim` must be positive: " + dim);
            }
        }

        this.dim = dim;
        return cl;
    }

    public void close() throws HiveException {
        if (model != null) {
            forwardModel();
            model = null;
            word2index = null;
        }
    }

    private void forwardModel() throws HiveException {
        final Text word = new Text();
        final IntWritable dimIndex = new IntWritable();
        final FloatWritable value = new FloatWritable();

        final Object[] result = new Object[3];
        result[0] = word;
        result[1] = dimIndex;
        result[2] = value;

        for (Map.Entry<String, Integer> entry : word2index.entrySet()) {
            int wordId = entry.getValue();
            word.set(entry.getKey());

            for (int i = 0; i < dim; i++) {
                dimIndex.set(i);
                value.set(model.inputWeights.get(wordId * dim + i));
                forward(result);
            }
        }
    }

    protected int getWordId(String word) {
        if (word2index.containsKey(word)) {
            return word2index.get(word);
        } else {
            int w = word2index.size();
            word2index.put(word, w);
            return w;
        }
    }

    protected abstract AbstractWord2VecModel createModel();
}
