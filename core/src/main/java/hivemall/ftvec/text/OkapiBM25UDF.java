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
package hivemall.ftvec.text;

import hivemall.UDFWithOptions;
import hivemall.utils.lang.mutable.MutableInt;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("deprecation")
@Description(name = "okapi_bm25",
        value = "_FUNC_(float tf_word, int dl, float avgdl, int N, int n [, const string options]) - Return an Okapi BM25 score in float")
public final class OkapiBM25UDF extends UDFWithOptions {

    public OkapiBM25UDF() {}

    @Nonnull
    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("tf_word", "termFrequencyOfWordInDoc", false,
                "Term frequency of a word in a document");
        opts.addOption("dl", "docLength", false, "Length of document in words");
        opts.addOption("avgdl", "averageDocLength", false, "Average length of documents in words");
        opts.addOption("N", "numDocs", false, "Number of documents");
        opts.addOption("n", "numDocsWithWord", false, "Number of documents containing the word q_i");
        opts.addOption("k_1", "k_1", true, "Hyperparameter usually in range 1.2 and 2.0 [default: 1.2]");
        opts.addOption("b", "b", true, "Hyperparameter [default: 0.75]");
        return opts;
    }

    @Nonnull
    @Override
    protected CommandLine processOptions(@Nonnull String opts) throws UDFArgumentException {
        return null;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] objectInspectors) throws UDFArgumentException {
        return null;
    }

    @Override
    public Object evaluate(DeferredObject[] deferredObjects) throws HiveException {
        return null;
    }

    @Override
    public String getDisplayString(String[] strings) {
        return null;
    }
}
