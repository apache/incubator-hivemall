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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.StringUtils;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "bm25",
        value = "_FUNC_(double termFrequency, int docLength, double avgDocLength, int numDocs, int numDocsWithTerm [, const string options]) "
                + "- Return an Okapi BM25 score in double. "
                + "Refer http://hivemall.incubator.apache.org/userguide/ft_engineering/bm25.html for usage")
@UDFType(deterministic = true, stateful = false)
public final class OkapiBM25UDF extends UDFWithOptions {

    private double k1 = 1.2d;
    private double b = 0.75d;

    // BM25+ https://en.wikipedia.org/wiki/Okapi_BM25#General_references
    private double delta = 0.d;

    // epsilon in https://en.wikipedia.org/wiki/Okapi_BM25#The_ranking_function
    private double minIDF = 1e-8;

    private PrimitiveObjectInspector frequencyOI;
    private PrimitiveObjectInspector docLengthOI;
    private PrimitiveObjectInspector averageDocLengthOI;
    private PrimitiveObjectInspector numDocsOI;
    private PrimitiveObjectInspector numDocsWithTermOI;

    @Nonnull
    private final DoubleWritable result = new DoubleWritable();

    public OkapiBM25UDF() {}

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k1", true,
            "Hyperparameter with type double, usually in range 1.2 and 2.0 [default: 1.2]");
        opts.addOption("b", true,
            "Hyperparameter with type double in range 0.0 and 1.0 [default: 0.75]");
        opts.addOption("d", "delta", true, "Hyperparameter delta of BM25+ [default: 0.0]");
        opts.addOption("min_idf", "epsilon", true, "Hyperparameter delta of BM25+ [default: 1e-8]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String opts) throws UDFArgumentException {
        CommandLine cl = parseOptions(opts);

        this.k1 = Primitives.parseDouble(cl.getOptionValue("k1"), k1);

        if (Primitives.isFinite(k1) == false || k1 < 0.0) {
            throw new UDFArgumentException("k1 must be a non-negative finite value: " + k1);
        }

        this.b = Primitives.parseDouble(cl.getOptionValue("b"), b);
        if (Double.isNaN(b) || b < 0.0 || b > 1.0) {
            throw new UDFArgumentException(
                "b1 hyperparameter must be in the range [0.0, 1.0]: " + b);
        }

        this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), delta);
        if (Primitives.isFinite(delta) == false) {
            throw new UDFArgumentException("Delta must be a finite value: " + delta);
        }

        this.minIDF = Primitives.parseDouble(cl.getOptionValue("min_idf"), minIDF);
        if (minIDF < 0.d) {
            throw new UDFArgumentException("min_idf must not be negative value: " + minIDF);
        }

        return cl;
    }

    @Override
    public ObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        final int numArgOIs = argOIs.length;
        if (numArgOIs < 5) {
            showHelp("#arguments must be greater than or equal to 5: " + numArgOIs);
        } else if (numArgOIs == 6) {
            String opts = HiveUtils.getConstString(argOIs[5]);
            processOptions(opts);
        }

        this.frequencyOI = HiveUtils.asDoubleCompatibleOI(argOIs[0]);
        this.docLengthOI = HiveUtils.asIntegerOI(argOIs[1]);
        this.averageDocLengthOI = HiveUtils.asDoubleCompatibleOI(argOIs[2]);
        this.numDocsOI = HiveUtils.asIntegerOI(argOIs[3]);
        this.numDocsWithTermOI = HiveUtils.asIntegerOI(argOIs[4]);

        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }

    @Override
    public DoubleWritable evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        Object arg1 = arguments[1].get();
        Object arg2 = arguments[2].get();
        Object arg3 = arguments[3].get();
        Object arg4 = arguments[4].get();

        if (arg0 == null || arg1 == null || arg2 == null || arg3 == null || arg4 == null) {
            throw new UDFArgumentException("Required arguments cannot be null");
        }

        double frequency = PrimitiveObjectInspectorUtils.getDouble(arg0, frequencyOI);
        int docLength = PrimitiveObjectInspectorUtils.getInt(arg1, docLengthOI);
        double averageDocLength = PrimitiveObjectInspectorUtils.getDouble(arg2, averageDocLengthOI);
        int numDocs = PrimitiveObjectInspectorUtils.getInt(arg3, numDocsOI);
        int numDocsWithTerm = PrimitiveObjectInspectorUtils.getInt(arg4, numDocsWithTermOI);

        assumeFalse(frequency < 0, "#frequency must be positive");
        assumeFalse(docLength < 1, "#docLength must be greater than or equal to 1");
        assumeFalse(averageDocLength <= 0.0, "#averageDocLength must be positive");
        assumeFalse(numDocs < 1, "#numDocs must be greater than or equal to 1");
        assumeFalse(numDocsWithTerm < 1, "#numDocsWithTerm must be greater than or equal to 1");

        double v = bm25(frequency, docLength, averageDocLength, numDocs, numDocsWithTerm);
        result.set(v);
        return result;
    }

    private double bm25(final double tf, final int docLength, final double averageDocLength,
            final int numDocs, final int numDocsWithTerm) {
        double numerator = tf * (k1 + 1);
        double denominator = tf + k1 * (1 - b + b * docLength / averageDocLength);
        double idf = Math.max(minIDF, idf(numDocs, numDocsWithTerm));
        return idf * (numerator / denominator + delta);
    }

    private static double idf(final int numDocs, final int numDocsWithTerm) {
        return Math.log10(1.0d + (numDocs - numDocsWithTerm + 0.5d) / (numDocsWithTerm + 0.5d));
    }

    @Override
    public String getDisplayString(String[] children) {
        return "bm25(" + StringUtils.join(children, ',') + ")";
    }

}
