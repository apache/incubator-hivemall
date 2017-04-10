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
import org.apache.hadoop.io.Text;

@Description(
        name = "lda_predict",
        value = "_FUNC_(string word, float value, int label, float lambda[, const string options])"
                + " - Returns a list which consists of <int label, float prob>")
public final class OnlineLDAPredictUDAF extends AbstractGenericUDAFResolver {
    private static final Log logger = LogFactory.getLog(OnlineLDAPredictUDAF.class);

    private OnlineLDAPredictUDAF() {}

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
        private StandardListObjectInspector wcListOI;
        private StandardMapObjectInspector lambdaMapOI;
        private StandardListObjectInspector lambdaMapElemOI;

        public Evaluator() {}

        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("k", "topic", true, "The number of topics [default: 10]");
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

            if (argOIs.length == 5) {
                String rawArgs = HiveUtils.getConstString(argOIs[4]);
                cl = parseOptions(rawArgs);
                this.topic = Primitives.parseInt(cl.getOptionValue("topic"), 10);
                this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), 1.f / topic);
                this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), 1E-5d);
            } else {
                this.topic = 10;
                this.alpha = 1.f / topic;
                this.delta = 1E-5d;
            }

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
                this.wcListOI = ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector);
                this.lambdaMapElemOI = ObjectInspectorFactory.getStandardListObjectInspector(
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
                PrimitiveObjectInspectorFactory.writableStringObjectInspector));

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

            myAggr.iterate(new Text(word), value, label, lambda);
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
            // String objects are passed as org.apache.hadoop.io.Text
            List<Text> wcList = (List<Text>) wcListOI.getList(wcListObj);

            Object lambdaMapObj = internalMergeOI.getStructFieldData(partial, lambdaMapField);
            if (lambdaMapObj instanceof LazyBinaryMap) {
                lambdaMapObj = ((LazyBinaryMap) lambdaMapObj).getMap();
            }
            Map<Text, Object> lambdaMapUncastElems = (Map<Text, Object>) lambdaMapOI.getMap(lambdaMapObj);

            Map<Text, List<Float>> lambdaMap = new HashMap<Text, List<Float>>();
            for (Text key : lambdaMapUncastElems.keySet()) {
                Object lambdaMapElemObj = lambdaMapUncastElems.get(key);
                if (lambdaMapElemObj instanceof  LazyBinaryArray) {
                    lambdaMapElemObj = ((LazyBinaryArray) lambdaMapElemObj).getList();
                }
                lambdaMap.put(key, (List<Float>) lambdaMapElemOI.getList(lambdaMapElemObj));
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

        private List<Text> wcList;
        private Map<Text, List<Float>> lambdaMap;

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
            this.wcList = new ArrayList<Text>();
            this.lambdaMap = new HashMap<Text, List<Float>>();
        }

        void iterate(Text word, float value, int label, float lambda) {
            wcList.add(new Text(word.toString() + ":" + value));

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

        void merge(List<Text> o_wcList, Map<Text, List<Float>> o_lambdaMap) {
            wcList.addAll(o_wcList);

            for (Map.Entry<Text, List<Float>> e : o_lambdaMap.entrySet()) {
                Text o_word = e.getKey();
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

            for (Text word : lambdaMap.keySet()) {
                List<Float> lambda_word = lambdaMap.get(word);
                for (int k = 0; k < topic; k++) {
                    Object lambdaObj = lambda_word.get(k); // this should be FloatWritable
                    float lambda = ((FloatWritable) lambdaObj).get();
                    model.setLambda(word.toString(), k, lambda);
                }
            }

            String[] wcArray = new String[wcList.size()];
            for (int i = 0; i < wcArray.length; i++) {
                wcArray[i] = wcList.get(i).toString();
            }

            return model.getTopicDistribution(wcArray);
        }
    }

}
