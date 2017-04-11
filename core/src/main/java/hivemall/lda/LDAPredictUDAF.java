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
package hivemall.lda;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.CommandLineUtils;
import hivemall.utils.lang.Primitives;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.SortedMap;
import java.util.TreeMap;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryMap;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StandardListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardMapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;

@Description(
        name = "lda_predict",
        value = "_FUNC_(string word, float value, int label, float lambda[, const string options])"
                + " - Returns a list which consists of <int label, float prob>")
public final class LDAPredictUDAF extends AbstractGenericUDAFResolver {
    private static final Log logger = LogFactory.getLog(LDAPredictUDAF.class);

    @Override
    public Evaluator getEvaluator(TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 4 && typeInfo.length != 5) {
            throw new UDFArgumentLengthException(
                "Expected argument length is 4 or 5 but given argument length was " + typeInfo.length);
        }

        if (!HiveUtils.isStringTypeInfo(typeInfo[0])) {
            throw new UDFArgumentTypeException(0,
                "String type is expected for the first argument word: " + typeInfo[0].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[1])) {
            throw new UDFArgumentTypeException(1,
                "Number type is expected for the second argument value: " + typeInfo[1].getTypeName());
        }
        if (!HiveUtils.isIntegerTypeInfo(typeInfo[2])) {
            throw new UDFArgumentTypeException(2,
                "Integer type is expected for the third argument label: " + typeInfo[2].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[3])) {
            throw new UDFArgumentTypeException(3,
                "Number type is expected for the forth argument lambda: " + typeInfo[3].getTypeName());
        }

        if (typeInfo.length == 5) {
            if (!HiveUtils.isStringTypeInfo(typeInfo[4])) {
                throw new UDFArgumentTypeException(4,
                    "String type is expected for the fifth argument lambda: " + typeInfo[4].getTypeName());
            }
        }

        return new Evaluator();
    }

    public static class Evaluator extends GenericUDAFEvaluator {

        // input OI
        private PrimitiveObjectInspector wordOI;
        private PrimitiveObjectInspector valueOI;
        private PrimitiveObjectInspector labelOI;
        private PrimitiveObjectInspector lambdaOI;

        // Hyperparameters
        private int topic;
        private float alpha;
        private double delta;

        // merge OI
        private StructObjectInspector internalMergeOI;
        private StructField wcListField;
        private StructField lambdaMapField;
        private StructField topicOptionField;
        private StructField alphaOptionField;
        private StructField deltaOptionField;
        private PrimitiveObjectInspector wcListElemOI;
        private StandardListObjectInspector wcListOI;
        private StandardMapObjectInspector lambdaMapOI;
        private PrimitiveObjectInspector lambdaMapKeyOI;
        private StandardListObjectInspector lambdaMapValueOI;

        public Evaluator() {}

        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("k", "topic", true, "The number of topics [required]");
            opts.addOption("alpha", true, "The hyperparameter for theta [default: 1/k]");
            opts.addOption("delta", true, "Check convergence in the expectation step [default: 1E-5]");
            return opts;
        }

        @Nonnull
        protected final CommandLine parseOptions(String optionValue) throws UDFArgumentException {
            String[] args = optionValue.split("\\s+");
            Options opts = getOptions();
            opts.addOption("help", false, "Show function help");
            CommandLine cl = CommandLineUtils.parseOptions(args, opts);

            if (cl.hasOption("help")) {
                Description funcDesc = getClass().getAnnotation(Description.class);
                final String cmdLineSyntax;
                if (funcDesc == null) {
                    cmdLineSyntax = getClass().getSimpleName();
                } else {
                    String funcName = funcDesc.name();
                    cmdLineSyntax = funcName == null ? getClass().getSimpleName()
                            : funcDesc.value().replace("_FUNC_", funcDesc.name());
                }
                StringWriter sw = new StringWriter();
                sw.write('\n');
                PrintWriter pw = new PrintWriter(sw);
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp(pw, HelpFormatter.DEFAULT_WIDTH, cmdLineSyntax, null, opts,
                        HelpFormatter.DEFAULT_LEFT_PAD, HelpFormatter.DEFAULT_DESC_PAD, null, true);
                pw.flush();
                String helpMsg = sw.toString();
                throw new UDFArgumentException(helpMsg);
            }

            return cl;
        }

        protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
            CommandLine cl = null;

            if (argOIs.length != 5) {
                throw new UDFArgumentException("At least 1 option `-topic` MUST be specified");
            }

            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);

            this.topic = Primitives.parseInt(cl.getOptionValue("topic"), 0);
            if (topic < 1) {
                throw new UDFArgumentException("A positive integer MUST be set to an option `-topic`: " + topic);
            }

            this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), 1.f / topic);
            this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), 1E-5d);

            return cl;
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 4 || parameters.length == 5);
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                processOptions(parameters);
                this.wordOI = HiveUtils.asStringOI(parameters[0]);
                this.valueOI = HiveUtils.asDoubleCompatibleOI(parameters[1]);
                this.labelOI = HiveUtils.asIntegerOI(parameters[2]);
                this.lambdaOI = HiveUtils.asDoubleCompatibleOI(parameters[3]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.wcListField = soi.getStructFieldRef("wcList");
                this.lambdaMapField = soi.getStructFieldRef("lambdaMap");
                this.topicOptionField = soi.getStructFieldRef("topic");
                this.alphaOptionField = soi.getStructFieldRef("alpha");
                this.deltaOptionField = soi.getStructFieldRef("delta");
                this.wcListElemOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                this.wcListOI = ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector);
                this.lambdaMapKeyOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                this.lambdaMapValueOI = ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector);
                this.lambdaMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaFloatObjectInspector));
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {
                final ArrayList<String> fieldNames = new ArrayList<String>();
                final ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
                fieldNames.add("label");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
                fieldNames.add("probability");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

                outputOI = ObjectInspectorFactory.getStandardListObjectInspector(
                    ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs));
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("wcList");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector));

            fieldNames.add("lambdaMap");
            fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector)));

            fieldNames.add("topic");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

            fieldNames.add("alpha");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

            fieldNames.add("delta");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @Override
        public AggregationBuffer getNewAggregationBuffer() throws HiveException {
            AggregationBuffer myAggr = new OnlineLDAPredictAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg) throws HiveException {
            OnlineLDAPredictAggregationBuffer myAggr = (OnlineLDAPredictAggregationBuffer) agg;
            myAggr.reset();
            myAggr.setOptions(topic, alpha, delta);
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg, Object[] parameters)
                throws HiveException {
            OnlineLDAPredictAggregationBuffer myAggr = (OnlineLDAPredictAggregationBuffer) agg;

            if (parameters[0] == null || parameters[1] == null || parameters[2] == null || parameters[3] == null) {
                return;
            }

            String word = PrimitiveObjectInspectorUtils.getString(parameters[0], wordOI);
            float value = HiveUtils.getFloat(parameters[1], valueOI);
            int label = PrimitiveObjectInspectorUtils.getInt(parameters[2], labelOI);
            float lambda = HiveUtils.getFloat(parameters[3], lambdaOI);

            myAggr.iterate(word, value, label, lambda);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            OnlineLDAPredictAggregationBuffer myAggr = (OnlineLDAPredictAggregationBuffer) agg;
            if(myAggr.wcList.size() == 0) {
                return null;
            }

            Object[] partialResult = new Object[5];
            partialResult[0] = myAggr.wcList;
            partialResult[1] = myAggr.lambdaMap;
            partialResult[2] = new IntWritable(myAggr.topic);
            partialResult[3] = new FloatWritable(myAggr.alpha);
            partialResult[4] = new DoubleWritable(myAggr.delta);

            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            Object wcListObj = internalMergeOI.getStructFieldData(partial, wcListField);

            if (wcListObj instanceof LazyBinaryArray) {
                wcListObj = ((LazyBinaryArray) wcListObj).getList();
            }
            List<?> wcListRaw = wcListOI.getList(wcListObj);

            // fix list elements to Java String objects
            int wcListSize = wcListRaw.size();
            List<String> wcList = new ArrayList<String>();
            for (int i = 0; i < wcListSize; i++) {
                wcList.add(PrimitiveObjectInspectorUtils.getString(wcListRaw.get(i), wcListElemOI));
            }

            Object lambdaMapObj = internalMergeOI.getStructFieldData(partial, lambdaMapField);
            if (lambdaMapObj instanceof LazyBinaryMap) {
                lambdaMapObj = ((LazyBinaryMap) lambdaMapObj).getMap();
            }
            Map<?, ?> lambdaMapRaw = lambdaMapOI.getMap(lambdaMapObj);

            Map<String, List<Float>> lambdaMap = new HashMap<String, List<Float>>();
            for (Map.Entry<?, ?> e : lambdaMapRaw.entrySet()) {
                // fix map keys to Java String objects
                String word = PrimitiveObjectInspectorUtils.getString(e.getKey(), lambdaMapKeyOI);

                Object lambdaMapValueObj = e.getValue();
                if (lambdaMapValueObj instanceof  LazyBinaryArray) {
                    lambdaMapValueObj = ((LazyBinaryArray) lambdaMapValueObj).getList();
                }
                List<?> lambdaMapValueRaw = lambdaMapValueOI.getList(lambdaMapValueObj);

                // fix map values to lists of Java Float objects
                int lambdaMapValueSize = lambdaMapValueRaw.size();
                List<Float> lambdaMapValue = new ArrayList<Float>();
                for (int i = 0; i < lambdaMapValueSize; i++) {
                    Object lambdaObj = lambdaMapValueRaw.get(i);
                    if (lambdaObj instanceof FloatWritable) {
                        lambdaMapValue.add(((FloatWritable) lambdaObj).get());
                    } else {
                        lambdaMapValue.add((Float) lambdaObj);
                    }
                }

                lambdaMap.put(word, lambdaMapValue);
            }

            // restore options from partial result
            Object topicObj = internalMergeOI.getStructFieldData(partial, topicOptionField);
            this.topic = PrimitiveObjectInspectorFactory.writableIntObjectInspector.get(topicObj);

            Object alphaObj = internalMergeOI.getStructFieldData(partial, alphaOptionField);
            this.alpha = PrimitiveObjectInspectorFactory.writableFloatObjectInspector.get(alphaObj);

            Object deltaObj = internalMergeOI.getStructFieldData(partial, deltaOptionField);
            this.delta = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(deltaObj);

            OnlineLDAPredictAggregationBuffer myAggr = (OnlineLDAPredictAggregationBuffer) agg;
            myAggr.setOptions(topic, alpha, delta);
            myAggr.merge(wcList, lambdaMap);
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            OnlineLDAPredictAggregationBuffer myAggr = (OnlineLDAPredictAggregationBuffer) agg;
            float[] topicDistr = myAggr.get();

            SortedMap<Float, Integer> sortedDistr = new TreeMap<Float, Integer>(Collections.reverseOrder());
            for (int i = 0; i < topicDistr.length; i++) {
                sortedDistr.put(topicDistr[i], i);
            }

            List<Object[]> result = new ArrayList<Object[]>();
            for (Map.Entry<Float, Integer> e : sortedDistr.entrySet()) {
                Object[] struct = new Object[2];
                struct[0] = new IntWritable(e.getValue()); // label
                struct[1] = new FloatWritable(e.getKey()); // probability
                result.add(struct);
            }
            return result;
        }

    }

    public static class OnlineLDAPredictAggregationBuffer extends GenericUDAFEvaluator.AbstractAggregationBuffer {

        private List<String> wcList;
        private Map<String, List<Float>> lambdaMap;

        private int topic;
        private float alpha;
        private double delta;

        OnlineLDAPredictAggregationBuffer() {
            super();
        }

        void setOptions(int topic, float alpha, double delta) {
            this.topic = topic;
            this.alpha = alpha;
            this.delta = delta;
        }

        void reset() {
            this.wcList = new ArrayList<String>();
            this.lambdaMap = new HashMap<String, List<Float>>();
        }

        void iterate(String word, float value, int label, float lambda) {
            wcList.add(word + ":" + value);

            // for an unforeseen word, initialize its lambdas w/ -1s
            if (!lambdaMap.containsKey(word)) {
                List<Float> lambdaEmpty_word = new ArrayList<Float>(Collections.nCopies(topic, -1.f));
                lambdaMap.put(word, lambdaEmpty_word);
            }

            // set the given lambda value
            List<Float> lambda_word = lambdaMap.get(word);
            lambda_word.set(label, lambda);
            lambdaMap.put(word, lambda_word);
        }

        void merge(List<String> o_wcList, Map<String, List<Float>> o_lambdaMap) {
            wcList.addAll(o_wcList);

            for (Map.Entry<String, List<Float>> e : o_lambdaMap.entrySet()) {
                String o_word = e.getKey();
                List<Float> o_lambda_word = e.getValue();

                if (!lambdaMap.containsKey(o_word)) { // for an unforeseen word
                    lambdaMap.put(o_word, o_lambda_word);
                } else { // for a partially observed word
                    List<Float> lambda_word = lambdaMap.get(o_word);
                    for (int k = 0; k < topic; k++) {
                        if (o_lambda_word.get(k) != -1.f) { // not default value
                            lambda_word.set(k, o_lambda_word.get(k)); // set the partial lambda value
                        }
                    }
                    lambdaMap.put(o_word, lambda_word);
                }
            }
        }

        float[] get() {
            OnlineLDAModel model = new OnlineLDAModel(topic, alpha, delta);

            for (String word : lambdaMap.keySet()) {
                List<Float> lambda_word = lambdaMap.get(word);
                for (int k = 0; k < topic; k++) {
                    model.setLambda(word, k, lambda_word.get(k));
                }
            }

            String[] wcArray = new String[wcList.size()];
            for (int i = 0; i < wcArray.length; i++) {
                wcArray[i] = wcList.get(i);
            }

            return model.getTopicDistribution(wcArray);
        }
    }

}
