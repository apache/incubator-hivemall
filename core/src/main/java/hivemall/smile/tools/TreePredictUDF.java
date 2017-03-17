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

import hivemall.smile.ModelType;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.regression.RegressionTree;
import hivemall.utils.codec.Base91;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.vector.DenseVector;
import hivemall.vector.SparseVector;
import hivemall.vector.Vector;

import java.io.IOException;
import java.util.Arrays;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;

@Description(
        name = "tree_predict",
        value = "_FUNC_(string modelId, int modelType, string script, array<double|string> features [, const boolean classification])"
                + " - Returns a prediction result of a random forest")
@UDFType(deterministic = true, stateful = false)
public final class TreePredictUDF extends GenericUDF {

    private boolean classification;
    private PrimitiveObjectInspector modelTypeOI;
    private StringObjectInspector stringOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private boolean denseInput;
    @Nullable
    private Vector featuresProbe;

    @Nullable
    private transient Evaluator evaluator;
    private boolean support_javascript_eval = true;

    @Override
    public void configure(MapredContext context) {
        super.configure(context);

        if (context != null) {
            JobConf conf = context.getJobConf();
            String tdJarVersion = conf.get("td.jar.version");
            if (tdJarVersion != null) {
                this.support_javascript_eval = false;
            }
        }
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 4 && argOIs.length != 5) {
            throw new UDFArgumentException("_FUNC_ takes 4 or 5 arguments");
        }

        this.modelTypeOI = HiveUtils.asIntegerOI(argOIs[1]);
        this.stringOI = HiveUtils.asStringOI(argOIs[2]);
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[3]);
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
                "_FUNC_ takes double[] or string[] for the first argument: " + listOI.getTypeName());
        }

        boolean classification = false;
        if (argOIs.length == 5) {
            classification = HiveUtils.getConstBoolean(argOIs[4]);
        }
        this.classification = classification;

        if (classification) {
            return PrimitiveObjectInspectorFactory.writableIntObjectInspector;
        } else {
            return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
        }
    }

    @Override
    public Writable evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            throw new HiveException("ModelId was null");
        }
        // Not using string OI for backward compatibilities
        String modelId = arg0.toString();

        Object arg1 = arguments[1].get();
        int modelTypeId = PrimitiveObjectInspectorUtils.getInt(arg1, modelTypeOI);
        ModelType modelType = ModelType.resolve(modelTypeId);

        Object arg2 = arguments[2].get();
        if (arg2 == null) {
            return null;
        }
        Text script = stringOI.getPrimitiveWritableObject(arg2);

        Object arg3 = arguments[3].get();
        if (arg3 == null) {
            throw new HiveException("array<double> features was null");
        }
        this.featuresProbe = parseFeatures(arg3, featuresProbe);

        if (evaluator == null) {
            this.evaluator = getEvaluator(modelType, support_javascript_eval);
        }

        Writable result = evaluator.evaluate(modelId, modelType.isCompressed(), script,
            featuresProbe, classification);
        return result;
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
                    throw new UDFArgumentException("Invaliad feature format `<index>:<value>`: "
                            + col);
                }

                final int colIndex = Integer.parseInt(feature);
                if (colIndex < 0) {
                    throw new UDFArgumentException(
                        "Col index MUST be greather than or equals to 0: " + colIndex);
                }
                probe.set(colIndex, value);
            }
        }
        return probe;
    }

    @Nonnull
    private static Evaluator getEvaluator(@Nonnull ModelType type, boolean supportJavascriptEval)
            throws UDFArgumentException {
        final Evaluator evaluator;
        switch (type) {
            case serialization:
            case serialization_compressed: {
                evaluator = new Evaluator();
                break;
            }
            case opscode:
            case opscode_compressed:
            case javascript:
            case javascript_compressed: {
                throw new UDFArgumentException("Deprecated model type `" + type
                        + "`. Please build models again.");
            }
            default:
                throw new UDFArgumentException("Unexpected model type was detected: " + type);
        }
        return evaluator;
    }

    @Override
    public void close() throws IOException {
        this.modelTypeOI = null;
        this.stringOI = null;
        this.featureElemOI = null;
        this.featureListOI = null;
        this.evaluator = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tree_predict(" + Arrays.toString(children) + ")";
    }

    static final class Evaluator {

        @Nullable
        private String prevModelId = null;
        private DecisionTree.Node cNode = null;
        private RegressionTree.Node rNode = null;

        Evaluator() {}

        public Writable evaluate(@Nonnull String modelId, boolean compressed, @Nonnull Text script,
                @Nonnull Vector features, boolean classification) throws HiveException {
            if (classification) {
                return evaluateClassification(modelId, compressed, script, features);
            } else {
                return evaluteRegression(modelId, compressed, script, features);
            }
        }

        private IntWritable evaluateClassification(@Nonnull String modelId, boolean compressed,
                @Nonnull Text script, @Nonnull Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.cNode = DecisionTree.deserializeNode(b, b.length, compressed);
            }
            assert (cNode != null);
            int result = cNode.predict(features);
            return new IntWritable(result);
        }

        private DoubleWritable evaluteRegression(@Nonnull String modelId, boolean compressed,
                @Nonnull Text script, @Nonnull Vector features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.rNode = RegressionTree.deserializeNode(b, b.length, compressed);
            }
            assert (rNode != null);
            double result = rNode.predict(features);
            return new DoubleWritable(result);
        }

    }

}
