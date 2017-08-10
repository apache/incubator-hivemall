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
package hivemall.tools.list;

import hivemall.utils.collections.BoundedPriorityQueue;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.CommandLineUtils;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.IntWritable;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.*;

/**
 * Return list of values sorted by value itself or specific key.
 */
@Description(name = "to_ordered_list",
        value = "_FUNC_(value, key [, const string options]) - Return list of values sorted by value itself or specific key")
public class UDAFToOrderedList extends AbstractGenericUDAFResolver {

    @SuppressWarnings("deprecation")
    @Override
    public GenericUDAFEvaluator getEvaluator(TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 2 && typeInfo.length != 3) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "Expecting two or three arguments: " + typeInfo.length);
        }
        if (typeInfo[1].getCategory() != ObjectInspector.Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(1,
                "Only primitive type arguments are accepted for the key but "
                        + typeInfo[1].getTypeName() + " was passed as the second parameter.");
        }
        return new UDAFToOrderedListEvaluator();
    }

    public static class UDAFToOrderedListEvaluator extends GenericUDAFEvaluator {

        private ObjectInspector valueOI;
        private PrimitiveObjectInspector keyOI;

        private ListObjectInspector valueListOI;
        private ListObjectInspector keyListOI;

        private StructObjectInspector internalMergeOI;

        private StructField valueListField;
        private StructField keyListField;
        private StructField sizeField;
        private StructField reverseOrderField;

        @Nonnegative
        private int size;
        private boolean reverseOrder;

        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("k", true, "To top-k (positive) or tail-k (negative) ordered queue");
            opts.addOption("reverse", "reverse_order", false,
                "Sort values by key in a reverse (e.g., descending) order [default: false]");
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

            int k = 0;
            boolean reverseOrder = false;

            if (argOIs.length >= 3) {
                String rawArgs = HiveUtils.getConstString(argOIs[2]);
                cl = parseOptions(rawArgs);

                reverseOrder = cl.hasOption("reverse_order");

                if (cl.hasOption("k")) {
                    k = Integer.parseInt(cl.getOptionValue("k"));
                    if (k == 0) {
                        throw new UDFArgumentException("`k` must be nonzero: " + k);
                    }
                }
            }

            this.size = Math.abs(k);

            if ((k > 0 && reverseOrder) || (k < 0 && !reverseOrder) || (k == 0 && !reverseOrder)) {
                // reverse top-k, natural tail-k = ascending = natural order output = reverse order priority queue
                this.reverseOrder = true;
            } else { // (k > 0 && !reverseOrder) || (k < 0 && reverseOrder) || (k == 0 && reverseOrder)
                // natural top-k or reverse tail-k = descending = reverse order output = natural order priority queue
                this.reverseOrder = false;
            }

            return cl;
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] argOIs) throws HiveException {
            super.init(mode, argOIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                processOptions(argOIs);
                this.valueOI = argOIs[0];
                this.keyOI = HiveUtils.asPrimitiveObjectInspector(argOIs[1]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) argOIs[0];
                this.internalMergeOI = soi;

                // re-extract input value OI
                this.valueListField = soi.getStructFieldRef("valueList");
                StandardListObjectInspector valueListOI = (StandardListObjectInspector) valueListField.getFieldObjectInspector();
                this.valueOI = valueListOI.getListElementObjectInspector();
                this.valueListOI = ObjectInspectorFactory.getStandardListObjectInspector(valueOI);

                // re-extract input key OI
                this.keyListField = soi.getStructFieldRef("keyList");
                StandardListObjectInspector keyListOI = (StandardListObjectInspector) keyListField.getFieldObjectInspector();
                this.keyOI = HiveUtils.asPrimitiveObjectInspector(keyListOI.getListElementObjectInspector());
                this.keyListOI = ObjectInspectorFactory.getStandardListObjectInspector(keyOI);

                this.sizeField = soi.getStructFieldRef("size");
                this.reverseOrderField = soi.getStructFieldRef("reverseOrder");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI(valueOI, keyOI);
            } else {// terminate
                outputOI = ObjectInspectorFactory.getStandardListObjectInspector(ObjectInspectorUtils.getStandardObjectInspector(valueOI));
            }

            return outputOI;
        }

        private static StructObjectInspector internalMergeOI(
                @Nonnull ObjectInspector valueOI, @Nonnull PrimitiveObjectInspector keyOI) {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("valueList");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(ObjectInspectorUtils.getStandardObjectInspector(valueOI)));

            fieldNames.add("keyList");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(ObjectInspectorUtils.getStandardObjectInspector(keyOI)));

            fieldNames.add("size");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

            fieldNames.add("reverseOrder");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableBooleanObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @SuppressWarnings("deprecation")
        @Override
        public AggregationBuffer getNewAggregationBuffer() throws HiveException {
            QueueAggregationBuffer myagg = new QueueAggregationBuffer();
            reset(myagg);
            return myagg;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;
            myagg.reset(size, reverseOrder);
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            if (parameters[0] == null || parameters[1] == null) {
                return;
            }

            Object value = ObjectInspectorUtils.copyToStandardObject(parameters[0], valueOI);
            Object key = ObjectInspectorUtils.copyToStandardObject(parameters[1], keyOI);

            TupleWithKey tuple = new TupleWithKey(key, value);
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;

            myagg.iterate(tuple);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;

            Map<String, List<Object>> tuples = myagg.drainQueue();
            List<Object> valueList = tuples.get("value");
            List<Object> keyList = tuples.get("key");
            if (valueList.size() == 0) {
                return null;
            }

            Object[] partialResult = new Object[4];
            partialResult[0] = valueList;
            partialResult[1] = keyList;
            partialResult[2] = new IntWritable(myagg.size);
            partialResult[3] = new BooleanWritable(myagg.reverseOrder);

            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            Object valueListObj = internalMergeOI.getStructFieldData(partial, valueListField);
            final List<?> valueListRaw = valueListOI.getList(HiveUtils.castLazyBinaryObject(valueListObj));
            final List<Object> valueList = new ArrayList<Object>();
            for (int i = 0, n = valueListRaw.size(); i < n; i++) {
                valueList.add(ObjectInspectorUtils.copyToStandardObject(valueListRaw.get(i),
                    valueOI));
            }

            Object keyListObj = internalMergeOI.getStructFieldData(partial, keyListField);
            final List<?> keyListRaw = keyListOI.getList(HiveUtils.castLazyBinaryObject(keyListObj));
            final List<Object> keyList = new ArrayList<Object>();
            for (int i = 0, n = keyListRaw.size(); i < n; i++) {
                keyList.add(ObjectInspectorUtils.copyToStandardObject(keyListRaw.get(i), keyOI));
            }

            Object sizeObj = internalMergeOI.getStructFieldData(partial, sizeField);
            int size = PrimitiveObjectInspectorFactory.writableIntObjectInspector.get(sizeObj);

            Object reverseOrderObj = internalMergeOI.getStructFieldData(partial, reverseOrderField);
            boolean reverseOrder = PrimitiveObjectInspectorFactory.writableBooleanObjectInspector.get(reverseOrderObj);

            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;
            myagg.setOptions(size, reverseOrder);
            myagg.merge(keyList, valueList);
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;
            Map<String, List<Object>> tuples = myagg.drainQueue();
            return tuples.get("value");
        }

        static class QueueAggregationBuffer extends AbstractAggregationBuffer {

            private AbstractQueueHandler queueHandler;

            @Nonnegative
            private int size;
            private boolean reverseOrder;

            QueueAggregationBuffer() {
                super();
            }

            void reset(@Nonnegative int size, boolean reverseOrder) {
                setOptions(size, reverseOrder);
                this.queueHandler = null;
            }

            void setOptions(@Nonnegative int size, boolean reverseOrder) {
                this.size = size;
                this.reverseOrder = reverseOrder;
            }

            void iterate(TupleWithKey tuple) {
                if (queueHandler == null) {
                    initQueueHandler();
                }
                queueHandler.offer(tuple);
            }

            void merge(List<Object> o_keyList, List<Object> o_valueList) {
                if (queueHandler == null) {
                    initQueueHandler();
                }
                for (int i = 0, n = o_keyList.size(); i < n; i++) {
                    queueHandler.offer(new TupleWithKey(o_keyList.get(i), o_valueList.get(i)));
                }
            }

            @Nonnull
            Map<String, List<Object>> drainQueue() {
                int n = queueHandler.size();
                final Object[] keys = new Object[n];
                final Object[] values = new Object[n];
                for (int i = n - 1; i >= 0; i--) { // head element in queue should be stored to tail of array
                    TupleWithKey tuple = queueHandler.poll();
                    keys[i] = tuple.getKey();
                    values[i] = tuple.getValue();
                }
                queueHandler.clear();

                Map<String, List<Object>> res = new HashMap<String, List<Object>>();
                res.put("key", Arrays.asList(keys));
                res.put("value", Arrays.asList(values));
                return res;
            }

            private void initQueueHandler() {
                final Comparator<TupleWithKey> comparator;
                if (reverseOrder) {
                    comparator = Collections.reverseOrder();
                } else {
                    comparator = new Comparator<TupleWithKey>() {
                        @Override
                        public int compare(TupleWithKey o1, TupleWithKey o2) {
                            return o1.compareTo(o2);
                        }
                    };
                }

                if (size > 0) {
                    this.queueHandler = new BoundedQueueHandler(size, comparator);
                } else {
                    this.queueHandler = new QueueHandler(comparator);
                }
            }

        }

        /**
         * Since BoundedPriorityQueue does not directly inherit PriorityQueue, we provide handler
         * class which wraps each of PriorityQueue and BoundedPriorityQueue.
         */
        private static abstract class AbstractQueueHandler {

            abstract void offer(TupleWithKey tuple);

            abstract int size();

            abstract TupleWithKey poll();

            abstract void clear();

        }

        private static final class QueueHandler extends AbstractQueueHandler {

            private static final int DEFAULT_INITIAL_CAPACITY = 11; // same as PriorityQueue

            private final PriorityQueue<TupleWithKey> queue;

            QueueHandler(@Nonnull Comparator<TupleWithKey> comparator) {
                this.queue = new PriorityQueue<TupleWithKey>(DEFAULT_INITIAL_CAPACITY, comparator);
            }

            @Override
            void offer(TupleWithKey tuple) {
                queue.offer(tuple);
            }

            @Override
            int size() {
                return queue.size();
            }

            @Override
            TupleWithKey poll() {
                return queue.poll();
            }

            @Override
            void clear() {
                queue.clear();
            }

        }

        private static final class BoundedQueueHandler extends AbstractQueueHandler {

            private final BoundedPriorityQueue<TupleWithKey> queue;

            BoundedQueueHandler(int size, @Nonnull Comparator<TupleWithKey> comparator) {
                this.queue = new BoundedPriorityQueue<TupleWithKey>(size, comparator);
            }

            @Override
            void offer(TupleWithKey tuple) {
                queue.offer(tuple);
            }

            @Override
            int size() {
                return queue.size();
            }

            @Override
            TupleWithKey poll() {
                return queue.poll();
            }

            @Override
            void clear() {
                queue.clear();
            }

        }

        private static final class TupleWithKey implements Comparable<TupleWithKey> {
            private Object key;
            private Object value;

            TupleWithKey(Object key, Object value) {
                this.key = key;
                this.value = value;
            }

            Object getKey() {
                return key;
            }

            Object getValue() {
                return value;
            }

            @Override
            public int compareTo(TupleWithKey o) {
                Comparable<? super Object> k = (Comparable<? super Object>) key;
                return k.compareTo(o.getKey());
            }
        }
    }
}
