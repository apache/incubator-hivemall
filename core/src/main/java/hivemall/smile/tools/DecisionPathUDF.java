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
package hivemall.smile.tools;

import hivemall.UDFWithOptions;
import matrix4j.vector.DenseVector;
import matrix4j.vector.SparseVector;
import matrix4j.vector.Vector;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.classification.PredictionHandler;
import hivemall.smile.regression.RegressionTree;
import hivemall.utils.codec.Base91;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.Text;

// @formatter:off
@Description(name = "decision_path",
        value = "_FUNC_(string modelId, string model, array<double|string> features [, const string options] [, optional array<string> featureNames=null, optional array<string> classNames=null])"
                + " - Returns a decision path for each prediction in array<string>",
        extended = "SELECT\n" + 
                "  t.passengerid,\n" + 
                "  decision_path(m.model_id, m.model, t.features, '-classification')\n" + 
                "FROM\n" + 
                "  model_rf m\n" + 
                "  LEFT OUTER JOIN\n" + 
                "  test_rf t;\n" +
                " | 892 | [\"2 [0.0] = 0.0\",\"0 [3.0] = 3.0\",\"1 [696.0] != 107.0\",\"7 [7.8292] <= 7.9104\",\"1 [696.0] != 828.0\",\"1 [696.0] != 391.0\",\"0 [0.961038961038961, 0.03896103896103896]\"] |\n\n" +
                "-- Show 100 frequent branches\n" +
                "WITH tmp as (\n" + 
                "  SELECT\n" + 
                "    decision_path(m.model_id, m.model, t.features, '-classification -no_verbose -no_leaf', array('pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked'), array('no','yes')) as path\n" + 
                "  FROM\n" + 
                "    model_rf m\n" + 
                "    LEFT OUTER JOIN -- CROSS JOIN\n" + 
                "    test_rf t\n" + 
                ")\n" + 
                "select\n" + 
                "  r.branch,\n" + 
                "  count(1) as cnt\n" + 
                "from\n" + 
                "  tmp l\n" + 
                "  LATERAL VIEW explode(l.path) r as branch\n" + 
                "group by\n" + 
                "  r.branch\n" + 
                "order by\n" + 
                "  cnt desc\n" + 
                "limit 100;")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class DecisionPathUDF extends UDFWithOptions {

    private StringObjectInspector modelOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private boolean denseInput;

    // options
    private boolean classification = false;
    private boolean summarize = true;
    private boolean verbose = true;
    private boolean noLeaf = false;

    @Nullable
    private String[] featureNames;
    @Nullable
    private String[] classNames;

    @Nullable
    private transient Vector featuresProbe;

    @Nullable
    private transient Evaluator evaluator;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("c", "classification", false,
            "Predict as classification [default: not enabled]");
        opts.addOption("no_sumarize", "disable_summarization", false,
            "Do not summarize decision paths");
        opts.addOption("no_verbose", "disable_verbose_output", false,
            "Disable verbose output [default: verbose]");
        opts.addOption("no_leaf", "disable_leaf_output", false,
            "Show leaf value [default: not enabled]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);

        this.classification = cl.hasOption("classification");
        this.summarize = !cl.hasOption("no_sumarize");
        this.verbose = !cl.hasOption("disable_verbose_output");
        this.noLeaf = cl.hasOption("disable_leaf_output");

        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 3 || argOIs.length > 6) {
            showHelp("decision_path takes 3 ~ 6 arguments");
        }

        this.modelOI = HiveUtils.asStringOI(argOIs[1]);

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[2]);
        this.featureListOI = listOI;
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.denseInput = true;
        } else if (HiveUtils.isStringOI(elemOI)) {
            this.featureElemOI = HiveUtils.asStringOI(elemOI);
            this.denseInput = false;
        } else {
            throw new UDFArgumentException(
                "tree_predict takes array<double> or array<string> for the 3rd argument: "
                        + listOI.getTypeName());
        }

        if (argOIs.length >= 4) {
            ObjectInspector argOI3 = argOIs[3];
            if (HiveUtils.isConstString(argOI3)) {
                String opts = HiveUtils.getConstString(argOI3);
                processOptions(opts);
                if (argOIs.length >= 5) {
                    ObjectInspector argOI4 = argOIs[4];
                    if (HiveUtils.isConstStringListOI(argOI4)) {
                        this.featureNames = HiveUtils.getConstStringArray(argOI4);
                        if (argOIs.length >= 6) {
                            ObjectInspector argOI5 = argOIs[5];
                            if (HiveUtils.isConstStringListOI(argOI5)) {
                                if (!classification) {
                                    throw new UDFArgumentException(
                                        "classNames should not be provided for regression");
                                }
                                this.classNames = HiveUtils.getConstStringArray(argOI5);
                            } else {
                                throw new UDFArgumentException(
                                    "decision_path expects 'const array<string> classNames' for the 6th argument: "
                                            + argOI5.getTypeName());
                            }
                        }
                    } else {
                        throw new UDFArgumentException(
                            "decision_path expects 'const array<string> featureNames' for the 5th argument: "
                                    + argOI4.getTypeName());
                    }
                }
            } else if (HiveUtils.isConstStringListOI(argOI3)) {
                this.featureNames = HiveUtils.getConstStringArray(argOI3);
                if (argOIs.length >= 5) {
                    ObjectInspector argOI4 = argOIs[4];
                    if (HiveUtils.isConstStringListOI(argOI4)) {
                        if (!classification) {
                            throw new UDFArgumentException(
                                "classNames should not be provided for regression");
                        }
                        this.classNames = HiveUtils.getConstStringArray(argOI4);
                    } else {
                        throw new UDFArgumentException(
                            "decision_path expects 'const array<string> classNames' for the 5th argument: "
                                    + argOI4.getTypeName());
                    }
                }
            } else {
                throw new UDFArgumentException(
                    "decision_path expects 'const array<string> options' or 'const array<string> featureNames' for the 4th argument: "
                            + argOI3.getTypeName());
            }
        }

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector);
    }

    @Override
    public List<String> evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            throw new HiveException("modelId should not be null");
        }
        // Not using string OI for backward compatibilities
        String modelId = arg0.toString();

        Object arg1 = arguments[1].get();
        if (arg1 == null) {
            return null;
        }
        Text model = modelOI.getPrimitiveWritableObject(arg1);

        Object arg2 = arguments[2].get();
        if (arg2 == null) {
            throw new HiveException("features was null");
        }
        this.featuresProbe = parseFeatures(arg2, featuresProbe);

        if (evaluator == null) {
            this.evaluator = classification ? new ClassificationEvaluator(this)
                    : new RegressionEvaluator(this);
        }
        return evaluator.evaluate(modelId, model, featuresProbe);
    }

    @Nonnull
    private Vector parseFeatures(@Nonnull final Object argObj, @Nullable Vector probe)
            throws UDFArgumentException {
        if (denseInput) {
            final int length = featureListOI.getListLength(argObj);
            if (probe == null) {
                probe = new DenseVector(length);
            } else if (length != probe.size()) {
                probe = new DenseVector(length);
            }

            for (int i = 0; i < length; i++) {
                final Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    probe.set(i, 0.d);
                } else {
                    double v = PrimitiveObjectInspectorUtils.getDouble(o, featureElemOI);
                    probe.set(i, v);
                }
            }
        } else {
            if (probe == null) {
                probe = new SparseVector();
            } else {
                probe.clear();
            }

            final int length = featureListOI.getListLength(argObj);
            for (int i = 0; i < length; i++) {
                Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    continue;
                }
                String col = o.toString();

                final int pos = col.indexOf(':');
                if (pos == 0) {
                    throw new UDFArgumentException("Invalid feature value representation: " + col);
                }

                final String feature;
                final double value;
                if (pos > 0) {
                    feature = col.substring(0, pos);
                    String s2 = col.substring(pos + 1);
                    value = Double.parseDouble(s2);
                } else {
                    feature = col;
                    value = 1.d;
                }

                if (feature.indexOf(':') != -1) {
                    throw new UDFArgumentException(
                        "Invalid feature format `<index>:<value>`: " + col);
                }

                final int colIndex = Integer.parseInt(feature);
                if (colIndex < 0) {
                    throw new UDFArgumentException(
                        "Col index MUST be greater than or equals to 0: " + colIndex);
                }
                probe.set(colIndex, value);
            }
        }
        return probe;
    }

    @Override
    public void close() throws IOException {
        this.modelOI = null;
        this.featureElemOI = null;
        this.featureListOI = null;
        this.featureNames = null;
        this.classNames = null;
        this.featuresProbe = null;
        this.evaluator = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "decision_path(" + StringUtils.join(children, ',') + ")";
    }

    interface Evaluator {

        @Nonnull
        List<String> evaluate(@Nonnull String modelId, @Nonnull Text model,
                @Nonnull Vector features) throws HiveException;

    }

    static final class ClassificationEvaluator implements Evaluator {

        @Nullable
        private final String[] featureNames;
        @Nullable
        private final String[] classNames;

        @Nonnull
        private final List<String> result;
        @Nonnull
        private final PredictionHandler handler;

        @Nullable
        private String prevModelId = null;
        private DecisionTree.Node cNode = null;

        ClassificationEvaluator(@Nonnull final DecisionPathUDF udf) {
            this.featureNames = udf.featureNames;
            this.classNames = udf.classNames;

            final StringBuilder buf = new StringBuilder();
            final ArrayList<String> result = new ArrayList<>();
            this.result = result;

            if (udf.summarize) {
                final LinkedHashMap<String, Double> map = new LinkedHashMap<>();

                this.handler = new PredictionHandler() {

                    @Override
                    public void init() {
                        map.clear();
                        result.clear();
                    }

                    @Override
                    public void visitBranch(Operator op, int splitFeatureIndex, double splitFeature,
                            double splitValue) {
                        buf.append(resolveFeatureName(splitFeatureIndex));
                        if (udf.verbose) {
                            buf.append(" [" + splitFeature + "] ");
                        } else {
                            buf.append(' ');
                        }
                        buf.append(op);
                        if (op == Operator.EQ || op == Operator.NE) {
                            buf.append(' ');
                            buf.append(splitValue);
                        }
                        String key = buf.toString();
                        map.put(key, splitValue);
                        StringUtils.clear(buf);
                    }

                    @Override
                    public void visitLeaf(int output, double[] posteriori) {
                        for (Map.Entry<String, Double> e : map.entrySet()) {
                            final String key = e.getKey();
                            if (key.indexOf('<') == -1 && key.indexOf('>') == -1) {
                                result.add(key);
                            } else {
                                double value = e.getValue().doubleValue();
                                result.add(key + ' ' + value);
                            }
                        }
                        if (udf.noLeaf) {
                            return;
                        }

                        if (udf.verbose) {
                            buf.append(resolveClassName(output));
                            buf.append(' ');
                            buf.append(Arrays.toString(posteriori));
                            result.add(buf.toString());
                            StringUtils.clear(buf);
                        } else {
                            result.add(resolveClassName(output));
                        }
                    }

                    @SuppressWarnings("unchecked")
                    @Override
                    public ArrayList<String> getResult() {
                        return result;
                    }

                };
            } else {
                this.handler = new PredictionHandler() {

                    @Override
                    public void init() {
                        result.clear();
                    }

                    @Override
                    public void visitBranch(Operator op, int splitFeatureIndex, double splitFeature,
                            double splitValue) {
                        buf.append(resolveFeatureName(splitFeatureIndex));
                        if (udf.verbose) {
                            buf.append(" [" + splitFeature + "] ");
                        } else {
                            buf.append(' ');
                        }
                        buf.append(op);
                        buf.append(' ');
                        buf.append(splitValue);
                        result.add(buf.toString());
                        StringUtils.clear(buf);
                    }

                    @Override
                    public void visitLeaf(int output, double[] posteriori) {
                        if (udf.noLeaf) {
                            return;
                        }

                        if (udf.verbose) {
                            buf.append(resolveClassName(output));
                            buf.append(' ');
                            buf.append(Arrays.toString(posteriori));
                            result.add(buf.toString());
                            StringUtils.clear(buf);
                        } else {
                            result.add(resolveClassName(output));
                        }
                    }

                    @SuppressWarnings("unchecked")
                    @Override
                    public ArrayList<String> getResult() {
                        return result;
                    }

                };
            }
        }

        @Nonnull
        private String resolveFeatureName(final int splitFeatureIndex) {
            if (featureNames == null) {
                return Integer.toString(splitFeatureIndex);
            } else {
                return featureNames[splitFeatureIndex];
            }
        }

        @Nonnull
        private String resolveClassName(final int classLabel) {
            if (classNames == null) {
                return Integer.toString(classLabel);
            } else {
                return classNames[classLabel];
            }
        }

        @Nonnull
        public List<String> evaluate(@Nonnull final String modelId, @Nonnull final Text script,
                @Nonnull final Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.cNode = DecisionTree.deserialize(b, b.length, true);
            }
            Preconditions.checkNotNull(cNode);

            handler.init();
            cNode.predict(features, handler);
            return handler.getResult();
        }

    }

    static final class RegressionEvaluator implements Evaluator {

        @Nullable
        private final String[] featureNames;

        @Nonnull
        private final List<String> result;
        @Nonnull
        private final PredictionHandler handler;

        @Nullable
        private String prevModelId = null;
        private RegressionTree.Node rNode = null;

        RegressionEvaluator(@Nonnull final DecisionPathUDF udf) {
            this.featureNames = udf.featureNames;

            final StringBuilder buf = new StringBuilder();
            final ArrayList<String> result = new ArrayList<>();
            this.result = result;

            if (udf.summarize) {
                final LinkedHashMap<String, Double> map = new LinkedHashMap<>();

                this.handler = new PredictionHandler() {

                    @Override
                    public void init() {
                        map.clear();
                        result.clear();
                    }

                    @Override
                    public void visitBranch(Operator op, int splitFeatureIndex, double splitFeature,
                            double splitValue) {
                        buf.append(resolveFeatureName(splitFeatureIndex));
                        if (udf.verbose) {
                            buf.append(" [" + splitFeature + "] ");
                        } else {
                            buf.append(' ');
                        }
                        buf.append(op);
                        if (op == Operator.EQ || op == Operator.NE) {
                            buf.append(' ');
                            buf.append(splitValue);
                        }
                        String key = buf.toString();
                        map.put(key, splitValue);
                        StringUtils.clear(buf);
                    }

                    @Override
                    public void visitLeaf(double output) {
                        for (Map.Entry<String, Double> e : map.entrySet()) {
                            final String key = e.getKey();
                            if (key.indexOf('<') == -1 && key.indexOf('>') == -1) {
                                result.add(key);
                            } else {
                                double value = e.getValue().doubleValue();
                                result.add(key + ' ' + value);
                            }
                        }
                        if (udf.noLeaf) {
                            return;
                        }

                        result.add(Double.toString(output));
                    }

                    @SuppressWarnings("unchecked")
                    @Override
                    public ArrayList<String> getResult() {
                        return result;
                    }

                };
            } else {
                this.handler = new PredictionHandler() {

                    @Override
                    public void init() {
                        result.clear();
                    }

                    @Override
                    public void visitBranch(Operator op, int splitFeatureIndex, double splitFeature,
                            double splitValue) {
                        buf.append(resolveFeatureName(splitFeatureIndex));
                        if (udf.verbose) {
                            buf.append(" [" + splitFeature + "] ");
                        }
                        buf.append(op);
                        buf.append(' ');
                        buf.append(splitValue);
                        result.add(buf.toString());
                        StringUtils.clear(buf);
                    }

                    @Override
                    public void visitLeaf(double output) {
                        if (udf.noLeaf) {
                            return;
                        }

                        result.add(Double.toString(output));
                    }

                    @SuppressWarnings("unchecked")
                    @Override
                    public ArrayList<String> getResult() {
                        return result;
                    }

                };
            }
        }

        @Nonnull
        private String resolveFeatureName(final int splitFeatureIndex) {
            if (featureNames == null) {
                return Integer.toString(splitFeatureIndex);
            } else {
                return featureNames[splitFeatureIndex];
            }
        }

        @Nonnull
        public List<String> evaluate(@Nonnull final String modelId, @Nonnull final Text script,
                @Nonnull final Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.rNode = RegressionTree.deserialize(b, b.length, true);
            }
            Preconditions.checkNotNull(rNode);

            handler.init();
            rNode.predict(features, handler);
            return handler.getResult();
        }
    }

}
