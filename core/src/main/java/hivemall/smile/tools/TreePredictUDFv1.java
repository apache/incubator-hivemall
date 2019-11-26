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

import static hivemall.smile.utils.SmileExtUtils.NUMERIC;

import hivemall.annotations.Since;
import hivemall.annotations.VisibleForTesting;
import hivemall.smile.vm.StackMachine;
import hivemall.smile.vm.VMRuntimeException;
import hivemall.utils.codec.Base91;
import hivemall.utils.codec.DeflateCodec;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.ObjectUtils;

import java.io.Closeable;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.script.Bindings;
import javax.script.Compilable;
import javax.script.CompiledScript;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

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

@Description(name = "tree_predict_v1",
        value = "_FUNC_(string modelId, int modelType, string script, array<double> features [, const boolean classification])"
                + " - Returns a prediction result of a random forest")
@UDFType(deterministic = true, stateful = false)
@Since(version = "v0.5-rc.1")
@Deprecated
public final class TreePredictUDFv1 extends GenericUDF {

    private boolean classification;
    private PrimitiveObjectInspector modelTypeOI;
    private StringObjectInspector stringOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;

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
            throw new UDFArgumentException("tree_predict_v1 takes 4 or 5 arguments");
        }

        this.modelTypeOI = HiveUtils.asIntegerOI(argOIs, 1);
        this.stringOI = HiveUtils.asStringOI(argOIs, 2);
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs, 3);
        this.featureListOI = listOI;
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);

        boolean classification = false;
        if (argOIs.length == 5) {
            classification = HiveUtils.getConstBoolean(argOIs, 4);
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
        double[] features = HiveUtils.asDoubleArray(arg3, featureListOI, featureElemOI);

        if (evaluator == null) {
            this.evaluator = getEvaluator(modelType, support_javascript_eval);
        }

        Writable result = evaluator.evaluate(modelId, modelType.isCompressed(), script, features,
            classification);
        return result;
    }

    @Nonnull
    private static Evaluator getEvaluator(@Nonnull ModelType type, boolean supportJavascriptEval)
            throws UDFArgumentException {
        final Evaluator evaluator;
        switch (type) {
            case serialization:
            case serialization_compressed: {
                evaluator = new JavaSerializationEvaluator();
                break;
            }
            case opscode:
            case opscode_compressed: {
                evaluator = new StackmachineEvaluator();
                break;
            }
            case javascript:
            case javascript_compressed: {
                if (!supportJavascriptEval) {
                    throw new UDFArgumentException(
                        "Javascript evaluation is not allowed in Treasure Data env");
                }
                evaluator = new JavascriptEvaluator();
                break;
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
        IOUtils.closeQuietly(evaluator);
        this.evaluator = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tree_predict(" + Arrays.toString(children) + ")";
    }

    enum ModelType {

        // not compressed
        opscode(1, false), javascript(2, false), serialization(3, false),
        // compressed
        opscode_compressed(-1, true), javascript_compressed(-2, true),
        serialization_compressed(-3, true);

        private final int id;
        private final boolean compressed;

        private ModelType(int id, boolean compressed) {
            this.id = id;
            this.compressed = compressed;
        }

        int getId() {
            return id;
        }

        boolean isCompressed() {
            return compressed;
        }

        @Nonnull
        static ModelType resolve(final int id) {
            final ModelType type;
            switch (id) {
                case 1:
                    type = opscode;
                    break;
                case -1:
                    type = opscode_compressed;
                    break;
                case 2:
                    type = javascript;
                    break;
                case -2:
                    type = javascript_compressed;
                    break;
                case 3:
                    type = serialization;
                    break;
                case -3:
                    type = serialization_compressed;
                    break;
                default:
                    throw new IllegalStateException("Unexpected ID for ModelType: " + id);
            }
            return type;
        }

    }

    public interface Evaluator extends Closeable {

        @Nullable
        Writable evaluate(@Nonnull String modelId, boolean compressed, @Nonnull final Text script,
                @Nonnull final double[] features, final boolean classification)
                throws HiveException;

    }

    static final class JavaSerializationEvaluator implements Evaluator {

        @Nullable
        private String prevModelId = null;
        private DtNodeV1 cNode = null;
        private RtNodeV1 rNode = null;

        JavaSerializationEvaluator() {}

        @Override
        public Writable evaluate(@Nonnull String modelId, boolean compressed, @Nonnull Text script,
                double[] features, boolean classification) throws HiveException {
            if (classification) {
                return evaluateClassification(modelId, compressed, script, features);
            } else {
                return evaluateRegression(modelId, compressed, script, features);
            }
        }

        private IntWritable evaluateClassification(@Nonnull String modelId, boolean compressed,
                @Nonnull Text script, double[] features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.cNode = deserializeDecisionTree(b, b.length, compressed);
            }
            assert (cNode != null);
            int result = cNode.predict(features);
            return new IntWritable(result);
        }

        @Nonnull
        @VisibleForTesting
        static DtNodeV1 deserializeDecisionTree(@Nonnull final byte[] serializedObj,
                final int length, final boolean compressed) throws HiveException {
            final DtNodeV1 root = new DtNodeV1();
            try {
                if (compressed) {
                    ObjectUtils.readCompressedObject(serializedObj, 0, length, root);
                } else {
                    ObjectUtils.readObject(serializedObj, length, root);
                }
            } catch (IOException ioe) {
                throw new HiveException("IOException cause while deserializing DecisionTree object",
                    ioe);
            } catch (Exception e) {
                throw new HiveException("Exception cause while deserializing DecisionTree object",
                    e);
            }
            return root;
        }

        private DoubleWritable evaluateRegression(@Nonnull String modelId, boolean compressed,
                @Nonnull Text script, double[] features) throws HiveException {
            if (!modelId.equals(prevModelId)) {
                this.prevModelId = modelId;
                int length = script.getLength();
                byte[] b = script.getBytes();
                b = Base91.decode(b, 0, length);
                this.rNode = deserializeRegressionTree(b, b.length, compressed);
            }
            assert (rNode != null);
            double result = rNode.predict(features);
            return new DoubleWritable(result);
        }

        @Nonnull
        @VisibleForTesting
        static RtNodeV1 deserializeRegressionTree(final byte[] serializedObj, final int length,
                final boolean compressed) throws HiveException {
            final RtNodeV1 root = new RtNodeV1();
            try {
                if (compressed) {
                    ObjectUtils.readCompressedObject(serializedObj, 0, length, root);
                } else {
                    ObjectUtils.readObject(serializedObj, length, root);
                }
            } catch (IOException ioe) {
                throw new HiveException("IOException cause while deserializing DecisionTree object",
                    ioe);
            } catch (Exception e) {
                throw new HiveException("Exception cause while deserializing DecisionTree object",
                    e);
            }
            return root;
        }

        @Override
        public void close() throws IOException {}

    }

    /**
     * Classification tree node.
     */
    static final class DtNodeV1 implements Externalizable {

        /**
         * Predicted class label for this node.
         */
        int output = -1;
        /**
         * The split feature for this node.
         */
        int splitFeature = -1;
        /**
         * The type of split feature
         */
        boolean quantitativeFeature = true;
        /**
         * The split value.
         */
        double splitValue = Double.NaN;
        /**
         * Reduction in splitting criterion.
         */
        double splitScore = 0.0;
        /**
         * Children node.
         */
        DtNodeV1 trueChild = null;
        /**
         * Children node.
         */
        DtNodeV1 falseChild = null;
        /**
         * Predicted output for children node.
         */
        int trueChildOutput = -1;
        /**
         * Predicted output for children node.
         */
        int falseChildOutput = -1;

        DtNodeV1() {}// for Externalizable

        /**
         * Constructor.
         */
        DtNodeV1(int output) {
            this.output = output;
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        int predict(final double[] x) {
            if (trueChild == null && falseChild == null) {
                return output;
            } else {
                if (quantitativeFeature) {
                    if (x[splitFeature] <= splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                } else {
                    if (x[splitFeature] == splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                }
            }
        }

        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
            this.output = in.readInt();
            this.splitFeature = in.readInt();
            int typeId = in.readInt();

            this.quantitativeFeature = (typeId == NUMERIC);
            this.splitValue = in.readDouble();
            if (in.readBoolean()) {
                this.trueChild = new DtNodeV1();
                trueChild.readExternal(in);
            }
            if (in.readBoolean()) {
                this.falseChild = new DtNodeV1();
                falseChild.readExternal(in);
            }
        }

    }

    /**
     * Regression tree node.
     */
    static final class RtNodeV1 implements Externalizable {

        /**
         * Predicted real value for this node.
         */
        double output = 0.0;
        /**
         * The split feature for this node.
         */
        int splitFeature = -1;
        /**
         * The type of split feature
         */
        boolean quantitativeFeature = true;
        /**
         * The split value.
         */
        double splitValue = Double.NaN;
        /**
         * Reduction in squared error compared to parent.
         */
        double splitScore = 0.0;
        /**
         * Children node.
         */
        RtNodeV1 trueChild;
        /**
         * Children node.
         */
        RtNodeV1 falseChild;
        /**
         * Predicted output for children node.
         */
        double trueChildOutput = 0.0;
        /**
         * Predicted output for children node.
         */
        double falseChildOutput = 0.0;

        RtNodeV1() {}//for Externalizable

        RtNodeV1(double output) {
            this.output = output;
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        double predict(final double[] x) {
            if (trueChild == null && falseChild == null) {
                return output;
            } else {
                if (quantitativeFeature) {
                    if (x[splitFeature] <= splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                } else {
                    // REVIEWME if(Math.equals(x[splitFeature], splitValue)) {
                    if (x[splitFeature] == splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                }
            }
        }

        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            throw new UnsupportedOperationException();
        }

        @Override
        public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
            this.output = in.readDouble();
            this.splitFeature = in.readInt();
            int typeId = in.readInt();
            this.quantitativeFeature = (typeId == NUMERIC);
            this.splitValue = in.readDouble();
            if (in.readBoolean()) {
                this.trueChild = new RtNodeV1();
                trueChild.readExternal(in);
            }
            if (in.readBoolean()) {
                this.falseChild = new RtNodeV1();
                falseChild.readExternal(in);
            }
        }
    }

    static final class StackmachineEvaluator implements Evaluator {

        private String prevModelId = null;
        private StackMachine prevVM = null;
        private DeflateCodec codec = null;

        StackmachineEvaluator() {}

        @Override
        public Writable evaluate(@Nonnull String modelId, boolean compressed, @Nonnull Text script,
                double[] features, boolean classification) throws HiveException {
            final String scriptStr;
            if (compressed) {
                if (codec == null) {
                    this.codec = new DeflateCodec(false, true);
                }
                byte[] b = script.getBytes();
                int len = script.getLength();
                b = Base91.decode(b, 0, len);
                try {
                    b = codec.decompress(b);
                } catch (IOException e) {
                    throw new HiveException("decompression failed", e);
                }
                scriptStr = new String(b);
            } else {
                scriptStr = script.toString();
            }

            final StackMachine vm;
            if (modelId.equals(prevModelId)) {
                vm = prevVM;
            } else {
                vm = new StackMachine();
                try {
                    vm.compile(scriptStr);
                } catch (VMRuntimeException e) {
                    throw new HiveException("failed to compile StackMachine", e);
                }
                this.prevModelId = modelId;
                this.prevVM = vm;
            }

            try {
                vm.eval(features);
            } catch (VMRuntimeException vme) {
                throw new HiveException("failed to eval StackMachine", vme);
            } catch (Throwable e) {
                throw new HiveException("failed to eval StackMachine", e);
            }

            Double result = vm.getResult();
            if (result == null) {
                return null;
            }
            if (classification) {
                return new IntWritable(result.intValue());
            } else {
                return new DoubleWritable(result.doubleValue());
            }
        }

        @Override
        public void close() throws IOException {
            IOUtils.closeQuietly(codec);
        }

    }

    static final class JavascriptEvaluator implements Evaluator {

        private final ScriptEngine scriptEngine;
        private final Compilable compilableEngine;

        private String prevModelId = null;
        private CompiledScript prevCompiled;

        private DeflateCodec codec = null;

        JavascriptEvaluator() throws UDFArgumentException {
            ScriptEngineManager manager = new ScriptEngineManager();
            ScriptEngine engine = manager.getEngineByExtension("js");
            if (!(engine instanceof Compilable)) {
                throw new UDFArgumentException(
                    "ScriptEngine was not compilable: " + engine.getFactory().getEngineName()
                            + " version " + engine.getFactory().getEngineVersion());
            }
            this.scriptEngine = engine;
            this.compilableEngine = (Compilable) engine;
        }

        @Override
        public Writable evaluate(@Nonnull String modelId, boolean compressed, @Nonnull Text script,
                double[] features, boolean classification) throws HiveException {
            final String scriptStr;
            if (compressed) {
                if (codec == null) {
                    this.codec = new DeflateCodec(false, true);
                }
                byte[] b = script.getBytes();
                int len = script.getLength();
                b = Base91.decode(b, 0, len);
                try {
                    b = codec.decompress(b);
                } catch (IOException e) {
                    throw new HiveException("decompression failed", e);
                }
                scriptStr = new String(b);
            } else {
                scriptStr = script.toString();
            }

            final CompiledScript compiled;
            if (modelId.equals(prevModelId)) {
                compiled = prevCompiled;
            } else {
                try {
                    compiled = compilableEngine.compile(scriptStr);
                } catch (ScriptException e) {
                    throw new HiveException("failed to compile: \n" + script, e);
                }
                this.prevCompiled = compiled;
            }

            final Bindings bindings = scriptEngine.createBindings();
            final Object result;
            try {
                bindings.put("x", features);
                result = compiled.eval(bindings);
            } catch (ScriptException se) {
                throw new HiveException("failed to evaluate: \n" + script, se);
            } catch (Throwable e) {
                throw new HiveException("failed to evaluate: \n" + script, e);
            } finally {
                bindings.clear();
            }

            if (result == null) {
                return null;
            }
            if (!(result instanceof Number)) {
                throw new HiveException("Got an unexpected non-number result: " + result);
            }
            if (classification) {
                Number casted = (Number) result;
                return new IntWritable(casted.intValue());
            } else {
                Number casted = (Number) result;
                return new DoubleWritable(casted.doubleValue());
            }
        }

        @Override
        public void close() throws IOException {
            IOUtils.closeQuietly(codec);
        }

    }

}
