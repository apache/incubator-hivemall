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
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import hivemall.utils.hadoop.HiveUtils;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;

import javax.annotation.Nonnull;
import java.util.Arrays;

@Description(name = "okapi_bm25",
        value = "_FUNC_(int termFrequency, int docLength, double avgDocLength, int numDocs, int numDocsWithTerm [, const string options]) - Return an Okapi BM25 score in double")
@UDFType(deterministic = true, stateful = false)
public final class OkapiBM25UDF extends UDFWithOptions {

    private static final double EPSILON = 1e-8;
    private static final double DEFAULT_K1 = 1.2;
    private static final double DEFAULT_B = 0.75;
    private double k1 = DEFAULT_K1;
    private double b = DEFAULT_B;
    private static final String K1_OPT_NAME = "k1";
    private static final String B_OPT_NAME = "b";

    private PrimitiveObjectInspector frequencyOI;
    private PrimitiveObjectInspector docLengthOI;
    private PrimitiveObjectInspector averageDocLengthOI;
    private PrimitiveObjectInspector numDocsOI;
    private PrimitiveObjectInspector numDocsWithTermOI;


    public OkapiBM25UDF() {}

    @Nonnull
    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("f", "frequencyOfTermInDoc", false, "Raw frequency of a word in a document");
        opts.addOption("dl", "docLength", false, "Length of document in words");
        opts.addOption("avgdl", "averageDocLength", false, "Average length of documents in words");
        opts.addOption("N", "numDocs", false, "Number of documents");
        opts.addOption("n", "numDocsWithTerm", false,
            "Number of documents containing the word q_i");
        opts.addOption(K1_OPT_NAME, "k1", true,
            "Hyperparameter with type double, usually in range 1.2 and 2.0 [default: 1.2]");
        opts.addOption(B_OPT_NAME, "b", true,
            "Hyperparameter with type double in range 0.0 and 1.0 [default: 0.75]");
        return opts;
    }

    @Nonnull
    @Override
    protected CommandLine processOptions(@Nonnull String opts) throws UDFArgumentException {
        CommandLine cl = parseOptions(opts);

        double k1Option =
                Double.parseDouble(cl.getOptionValue(K1_OPT_NAME, Double.toString(DEFAULT_K1)));
        if (k1Option < 0.0) {
            throw new UDFArgumentException(
                String.format("#%s hyperparameter must be positive", K1_OPT_NAME));
        }
        k1 = k1Option;

        double bOption =
                Double.parseDouble(cl.getOptionValue(B_OPT_NAME, Double.toString(DEFAULT_B)));
        if (bOption < 0.0 || bOption > 1.0) {
            throw new UDFArgumentException(
                String.format("#%s hyperparameter must be in the range [0.0, 1.0]", B_OPT_NAME));
        }
        b = bOption;

        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgOIs = argOIs.length;
        if (numArgOIs < 5) {
            throw new UDFArgumentException("argOIs.length must be greater than or equal to 5");
        } else if (numArgOIs == 6) {
            String opts = HiveUtils.getConstString(argOIs[5]);
            processOptions(opts);
        }

        this.frequencyOI = HiveUtils.asIntegerOI(argOIs[0]);
        this.docLengthOI = HiveUtils.asIntegerOI(argOIs[1]);
        this.averageDocLengthOI = HiveUtils.asDoubleOI(argOIs[2]);
        this.numDocsOI = HiveUtils.asIntegerOI(argOIs[3]);
        this.numDocsWithTermOI = HiveUtils.asIntegerOI(argOIs[4]);

        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }

    @Override
    public DoubleWritable evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        Object arg1 = arguments[1].get();
        Object arg2 = arguments[2].get();
        Object arg3 = arguments[3].get();
        Object arg4 = arguments[4].get();

        if (arg0 == null || arg1 == null || arg2 == null || arg3 == null || arg4 == null) {
            throw new UDFArgumentException("Required arguments cannot be null");
        }

        int frequency = PrimitiveObjectInspectorUtils.getInt(arg0, frequencyOI);
        int docLength = PrimitiveObjectInspectorUtils.getInt(arg1, docLengthOI);
        double averageDocLength = PrimitiveObjectInspectorUtils.getDouble(arg2, averageDocLengthOI);
        int numDocs = PrimitiveObjectInspectorUtils.getInt(arg3, numDocsOI);
        int numDocsWithTerm = PrimitiveObjectInspectorUtils.getInt(arg4, numDocsWithTermOI);

        if (frequency < 0) {
            throw new UDFArgumentException("#frequency must be positive");
        }

        if (docLength < 1) {
            throw new UDFArgumentException("#docLength must be greater than or equal to 1");
        }

        if (averageDocLength < 0.0) {
            throw new UDFArgumentException("#averageDocLength must be positive");
        }

        if (Math.abs(averageDocLength) < EPSILON) {
            throw new UDFArgumentException("#averageDocLength cannot be 0");
        }

        if (numDocs < 1) {
            throw new UDFArgumentException("#numDocs must be greater than or equal to 1");
        }

        if (numDocsWithTerm < 1) {
            throw new UDFArgumentException("#numDocsWithTerm must be greater than or equal to 1");
        }


        double result =
                calculateBM25(frequency, docLength, averageDocLength, numDocs, numDocsWithTerm);

        return new DoubleWritable(result);
    }

    private double calculateBM25(int frequency, int docLength, double averageDocLength, int numDocs,
            int numDocsWithTerm) {
        double numerator = frequency * (k1 + 1);
        double denominator = frequency + k1 * (1 - b + b * docLength / averageDocLength);
        double idf = calculateIDF(numDocs, numDocsWithTerm);
        return idf * numerator / denominator;
    }

    private static double calculateIDF(int numDocs, int numDocsWithTerm) {
        return Math.log10(1.0 + (numDocs - numDocsWithTerm + 0.5) / (numDocsWithTerm + 0.5));
    }

    @Override
    public String getDisplayString(String[] children) {
        return "okapi_bm25(" + Arrays.toString(children) + ")";
    }
}
