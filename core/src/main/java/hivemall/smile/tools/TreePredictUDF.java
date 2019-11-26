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
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

@Description(name = "tree_predict",
        value = "_FUNC_(string modelId, string model, array<double|string> features [, const string options | const boolean classification=false])"
                + " - Returns a prediction result of a random forest"
                + " in <int value, array<double> a posteriori> for classification and <double> for regression")
@UDFType(deterministic = true, stateful = false)
public final class TreePredictUDF extends UDFWithOptions {

    private boolean classification;
    private StringObjectInspector modelOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private boolean denseInput;
    @Nullable
    private Vector featuresProbe;

    @Nullable
    private transient Evaluator evaluator;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("c", "classification", false,
            "Predict as classification [default: not enabled]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);

        this.classification = cl.hasOption("classification");
        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 3 && argOIs.length != 4) {
            showHelp("tree_predict takes 3 or 4 arguments");
        }

        this.modelOI = HiveUtils.asStringOI(argOIs, 1);
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs, 2);
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
                "tree_predict takes array<double> or array<string> for the second argument: "
                        + listOI.getTypeName());
        }

        if (argOIs.length == 4) {
            ObjectInspector argOI3 = argOIs[3];
            if (HiveUtils.isConstBoolean(argOI3)) {
                this.classification = HiveUtils.getConstBoolean(argOI3);
            } else if (HiveUtils.isConstString(argOI3)) {
                String opts = HiveUtils.getConstString(argOI3);
                processOptions(opts);
            } else {
                throw new UDFArgumentException(
                    "tree_predict expects <const boolean> or <const string> for the fourth argument: "
                            + argOI3.getTypeName());
            }
        } else {
            this.classification = false;
        }

        if (classification) {
            List<String> fieldNames = new ArrayList<String>(2);
            List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(2);
            fieldNames.add("value");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
            fieldNames.add("posteriori");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        } else {
            return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
        }
    }

    @Override
    public Object evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
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
            this.evaluator =
                    classification ? new ClassificationEvaluator() : new RegressionEvaluator();
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
        this.evaluator = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tree_predict(" + Arrays.toString(children) + ")";
    }

    interface Evaluator {

        @Nonnull
        Object evaluate(@Nonnull String modelId, @Nonnull Text model, @Nonnull Vector features)
                throws HiveException;

    }

    static final class ClassificationEvaluator implements Evaluator {

        @Nonnull
        private final Object[] result;

        @Nullable
        private String prevModelId = null;
        private DecisionTree.Node cNode = null;

        ClassificationEvaluator() {
            this.result = new Object[2];
        }

        @Nonnull
        public Object[] evaluate(@Nonnull final String modelId, @Nonnull final Text script,
                @Nonnull final Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.cNode = DecisionTree.deserialize(b, b.length, true);
            }

            Arrays.fill(result, null);
            Preconditions.checkNotNull(cNode);
            cNode.predict(features, new PredictionHandler() {
                public void visitLeaf(int output, double[] posteriori) {
                    result[0] = new IntWritable(output);
                    result[1] = WritableUtils.toWritableList(posteriori);
                }
            });

            return result;
        }

    }

    static final class RegressionEvaluator implements Evaluator {

        @Nonnull
        private final DoubleWritable result;

        @Nullable
        private String prevModelId = null;
        private RegressionTree.Node rNode = null;

        RegressionEvaluator() {
            this.result = new DoubleWritable();
        }

        @Nonnull
        public DoubleWritable evaluate(@Nonnull final String modelId, @Nonnull final Text script,
                @Nonnull final Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.rNode = RegressionTree.deserialize(b, b.length, true);
            }
            Preconditions.checkNotNull(rNode);

            double value = rNode.predict(features);
            result.set(value);
            return result;
        }
    }

}
