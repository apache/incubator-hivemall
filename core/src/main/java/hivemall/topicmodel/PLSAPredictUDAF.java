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
package hivemall.topicmodel;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.CommandLineUtils;
import hivemall.utils.lang.Primitives;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

@Description(name = "plsa_predict",
        value = "_FUNC_(string word, float value, int label, float prob[, const string options])"
                + " - Returns a list which consists of <int label, float prob>")
public final class PLSAPredictUDAF extends AbstractGenericUDAFResolver {
    private static final Log logger = LogFactory.getLog(PLSAPredictUDAF.class);

    @Override
    public Evaluator getEvaluator(TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 4 && typeInfo.length != 5) {
            throw new UDFArgumentLengthException(
                "Expected argument length is 4 or 5 but given argument length was "
                        + typeInfo.length);
        }

        if (!HiveUtils.isStringTypeInfo(typeInfo[0])) {
            throw new UDFArgumentTypeException(0,
                "String type is expected for the first argument word: " + typeInfo[0].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[1])) {
            throw new UDFArgumentTypeException(1,
                "Number type is expected for the second argument value: "
                        + typeInfo[1].getTypeName());
        }
        if (!HiveUtils.isIntegerTypeInfo(typeInfo[2])) {
            throw new UDFArgumentTypeException(2,
                "Integer type is expected for the third argument label: "
                        + typeInfo[2].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[3])) {
            throw new UDFArgumentTypeException(3,
                "Number type is expected for the forth argument prob: " + typeInfo[3].getTypeName());
        }

        if (typeInfo.length == 5) {
            if (!HiveUtils.isStringTypeInfo(typeInfo[4])) {
                throw new UDFArgumentTypeException(4,
                    "String type is expected for the fifth argument prob: "
                            + typeInfo[4].getTypeName());
            }
        }

        return new Evaluator();
    }

    public static class Evaluator extends GenericUDAFEvaluator {

        // input OI
        private PrimitiveObjectInspector wordOI;
        private PrimitiveObjectInspector valueOI;
        private PrimitiveObjectInspector labelOI;
        private PrimitiveObjectInspector probOI;

        // Hyperparameters
        private int topic;
        private float alpha;
        private double delta;

        // merge OI
        private StructObjectInspector internalMergeOI;
        private StructField wcListField;
        private StructField probMapField;
        private StructField topicOptionField;
        private StructField alphaOptionField;
        private StructField deltaOptionField;
        private PrimitiveObjectInspector wcListElemOI;
        private StandardListObjectInspector wcListOI;
        private StandardMapObjectInspector probMapOI;
        private PrimitiveObjectInspector probMapKeyOI;
        private StandardListObjectInspector probMapValueOI;
        private PrimitiveObjectInspector probMapValueElemOI;

        public Evaluator() {}

        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("k", "topic", true, "The number of topics [required]");
            opts.addOption("alpha", true, "The hyperparameter for P(w|z) update [default: 0.5]");
            opts.addOption("delta", true,
                "Check convergence in the expectation step [default: 1E-5]");
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
                throw new UDFArgumentException(
                    "A positive integer MUST be set to an option `-topic`: " + topic);
            }

            this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), 0.5f);
            this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), 1E-3d);

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
                this.probOI = HiveUtils.asDoubleCompatibleOI(parameters[3]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.wcListField = soi.getStructFieldRef("wcList");
                this.probMapField = soi.getStructFieldRef("probMap");
                this.topicOptionField = soi.getStructFieldRef("topic");
                this.alphaOptionField = soi.getStructFieldRef("alpha");
                this.deltaOptionField = soi.getStructFieldRef("delta");
                this.wcListElemOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                this.wcListOI = ObjectInspectorFactory.getStandardListObjectInspector(wcListElemOI);
                this.probMapKeyOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                this.probMapValueElemOI = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                this.probMapValueOI = ObjectInspectorFactory.getStandardListObjectInspector(probMapValueElemOI);
                this.probMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(probMapKeyOI,
                    probMapValueOI);
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

                outputOI = ObjectInspectorFactory.getStandardListObjectInspector(ObjectInspectorFactory.getStandardStructObjectInspector(
                    fieldNames, fieldOIs));
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("wcList");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector));

            fieldNames.add("probMap");
            fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaFloatObjectInspector)));

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
            AggregationBuffer myAggr = new PLSAPredictAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            PLSAPredictAggregationBuffer myAggr = (PLSAPredictAggregationBuffer) agg;
            myAggr.reset();
            myAggr.setOptions(topic, alpha, delta);
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            PLSAPredictAggregationBuffer myAggr = (PLSAPredictAggregationBuffer) agg;

            if (parameters[0] == null || parameters[1] == null || parameters[2] == null
                    || parameters[3] == null) {
                return;
            }

            String word = PrimitiveObjectInspectorUtils.getString(parameters[0], wordOI);
            float value = HiveUtils.getFloat(parameters[1], valueOI);
            int label = PrimitiveObjectInspectorUtils.getInt(parameters[2], labelOI);
            float prob = HiveUtils.getFloat(parameters[3], probOI);

            myAggr.iterate(word, value, label, prob);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            PLSAPredictAggregationBuffer myAggr = (PLSAPredictAggregationBuffer) agg;
            if (myAggr.wcList.size() == 0) {
                return null;
            }

            Object[] partialResult = new Object[5];
            partialResult[0] = myAggr.wcList;
            partialResult[1] = myAggr.probMap;
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

            List<?> wcListRaw = wcListOI.getList(HiveUtils.castLazyBinaryObject(wcListObj));

            // fix list elements to Java String objects
            int wcListSize = wcListRaw.size();
            List<String> wcList = new ArrayList<String>();
            for (int i = 0; i < wcListSize; i++) {
                wcList.add(PrimitiveObjectInspectorUtils.getString(wcListRaw.get(i), wcListElemOI));
            }

            Object probMapObj = internalMergeOI.getStructFieldData(partial, probMapField);
            Map<?, ?> probMapRaw = probMapOI.getMap(HiveUtils.castLazyBinaryObject(probMapObj));

            Map<String, List<Float>> probMap = new HashMap<String, List<Float>>();
            for (Map.Entry<?, ?> e : probMapRaw.entrySet()) {
                // fix map keys to Java String objects
                String word = PrimitiveObjectInspectorUtils.getString(e.getKey(), probMapKeyOI);

                Object probMapValueObj = e.getValue();
                List<?> probMapValueRaw = probMapValueOI.getList(HiveUtils.castLazyBinaryObject(probMapValueObj));

                // fix map values to lists of Java Float objects
                int probMapValueSize = probMapValueRaw.size();
                List<Float> prob_word = new ArrayList<Float>();
                for (int i = 0; i < probMapValueSize; i++) {
                    prob_word.add(HiveUtils.getFloat(probMapValueRaw.get(i), probMapValueElemOI));
                }

                probMap.put(word, prob_word);
            }

            // restore options from partial result
            Object topicObj = internalMergeOI.getStructFieldData(partial, topicOptionField);
            this.topic = PrimitiveObjectInspectorFactory.writableIntObjectInspector.get(topicObj);

            Object alphaObj = internalMergeOI.getStructFieldData(partial, alphaOptionField);
            this.alpha = PrimitiveObjectInspectorFactory.writableFloatObjectInspector.get(alphaObj);

            Object deltaObj = internalMergeOI.getStructFieldData(partial, deltaOptionField);
            this.delta = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(deltaObj);

            PLSAPredictAggregationBuffer myAggr = (PLSAPredictAggregationBuffer) agg;
            myAggr.setOptions(topic, alpha, delta);
            myAggr.merge(wcList, probMap);
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            PLSAPredictAggregationBuffer myAggr = (PLSAPredictAggregationBuffer) agg;
            float[] topicDistr = myAggr.get();

            SortedMap<Float, Integer> sortedDistr = new TreeMap<Float, Integer>(
                Collections.reverseOrder());
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

    public static class PLSAPredictAggregationBuffer extends
            GenericUDAFEvaluator.AbstractAggregationBuffer {

        private List<String> wcList;
        private Map<String, List<Float>> probMap;

        private int topic;
        private float alpha;
        private double delta;

        PLSAPredictAggregationBuffer() {
            super();
        }

        void setOptions(int topic, float alpha, double delta) {
            this.topic = topic;
            this.alpha = alpha;
            this.delta = delta;
        }

        void reset() {
            this.wcList = new ArrayList<String>();
            this.probMap = new HashMap<String, List<Float>>();
        }

        void iterate(String word, float value, int label, float prob) {
            wcList.add(word + ":" + value);

            // for an unforeseen word, initialize its probs w/ -1s
            if (!probMap.containsKey(word)) {
                List<Float> probEmpty_word = new ArrayList<Float>(Collections.nCopies(topic, -1.f));
                probMap.put(word, probEmpty_word);
            }

            // set the given prob value
            List<Float> prob_word = probMap.get(word);
            prob_word.set(label, prob);
            probMap.put(word, prob_word);
        }

        void merge(List<String> o_wcList, Map<String, List<Float>> o_probMap) {
            wcList.addAll(o_wcList);

            for (Map.Entry<String, List<Float>> e : o_probMap.entrySet()) {
                String o_word = e.getKey();
                List<Float> o_prob_word = e.getValue();

                if (!probMap.containsKey(o_word)) { // for an unforeseen word
                    probMap.put(o_word, o_prob_word);
                } else { // for a partially observed word
                    List<Float> prob_word = probMap.get(o_word);
                    for (int k = 0; k < topic; k++) {
                        if (o_prob_word.get(k) != -1.f) { // not default value
                            prob_word.set(k, o_prob_word.get(k)); // set the partial prob value
                        }
                    }
                    probMap.put(o_word, prob_word);
                }
            }
        }

        float[] get() {
            IncrementalPLSAModel model = new IncrementalPLSAModel(topic, alpha, delta);

            for (String word : probMap.keySet()) {
                List<Float> prob_word = probMap.get(word);
                for (int k = 0; k < topic; k++) {
                    if (prob_word.get(k) != -1.f) {
                        model.setProbability(word, k, prob_word.get(k));
                    }
                }
            }

            String[] wcArray = wcList.toArray(new String[wcList.size()]);
            return model.getTopicDistribution(wcArray);
        }
    }

}
