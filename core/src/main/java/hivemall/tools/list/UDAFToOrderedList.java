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
import hivemall.utils.lang.NaturalComparator;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.struct.Pair;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

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
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.IntWritable;

/**
 * Return list of values sorted by value itself or specific key.
 */
@Description(name = "to_ordered_list",
        value = "_FUNC_(PRIMITIVE value [, PRIMITIVE key, const string options])"
                + " - Return list of values sorted by value itself or specific key")
public final class UDAFToOrderedList extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        @SuppressWarnings("deprecation")
        TypeInfo[] typeInfo = info.getParameters();
        ObjectInspector[] argOIs = info.getParameterObjectInspectors();
        if ((typeInfo.length == 1) || (typeInfo.length == 2 && HiveUtils.isConstString(argOIs[1]))) {
            // sort values by value itself w/o key
            if (typeInfo[0].getCategory() != ObjectInspector.Category.PRIMITIVE) {
                throw new UDFArgumentTypeException(0,
                    "Only primitive type arguments are accepted for value but "
                            + typeInfo[0].getTypeName() + " was passed as the first parameter.");
            }
        } else if ((typeInfo.length == 2)
                || (typeInfo.length == 3 && HiveUtils.isConstString(argOIs[2]))) {
            // sort values by key
            if (typeInfo[1].getCategory() != ObjectInspector.Category.PRIMITIVE) {
                throw new UDFArgumentTypeException(1,
                    "Only primitive type arguments are accepted for key but "
                            + typeInfo[1].getTypeName() + " was passed as the second parameter.");
            }
        } else {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "Number of arguments must be in [1, 3] including constant string for options: "
                        + typeInfo.length);
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
        private boolean sortByKey;

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

            int optionIndex = 1;
            if (sortByKey) {
                optionIndex = 2;
            }

            int k = 0;
            boolean reverseOrder = false;
            if (argOIs.length >= optionIndex + 1) {
                String rawArgs = HiveUtils.getConstString(argOIs[optionIndex]);
                cl = parseOptions(rawArgs);

                reverseOrder = cl.hasOption("reverse_order");

                if (cl.hasOption("k")) {
                    k = Integer.parseInt(cl.getOptionValue("k"));
                    if (k == 0) {
                        throw new UDFArgumentException("`k` must be non-zero value: " + k);
                    }
                }
            }
            this.size = Math.abs(k);

            if ((k > 0 && reverseOrder) || (k < 0 && reverseOrder == false)
                    || (k == 0 && reverseOrder == false)) {
                // top-k on reverse order = tail-k on natural order (so, top-k on descending)
                this.reverseOrder = true;
            } else { // (k > 0 && reverseOrder == false) || (k < 0 && reverseOrder) || (k == 0 && reverseOrder)
                // top-k on natural order = tail-k on reverse order (so, top-k on ascending)
                this.reverseOrder = false;
            }

            return cl;
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] argOIs) throws HiveException {
            super.init(mode, argOIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                // this flag will be used in `processOptions` and `iterate` (= when Mode.PARTIAL1 or Mode.COMPLETE)
                this.sortByKey = (argOIs.length == 2 && !HiveUtils.isConstString(argOIs[1]))
                        || (argOIs.length == 3 && HiveUtils.isConstString(argOIs[2]));

                if (sortByKey) {
                    this.valueOI = HiveUtils.asPrimitiveObjectInspector(argOIs[0]);
                    this.keyOI = HiveUtils.asPrimitiveObjectInspector(argOIs[1]);
                } else {
                    // sort values by value itself
                    this.valueOI = HiveUtils.asPrimitiveObjectInspector(argOIs[0]);
                    this.keyOI = HiveUtils.asPrimitiveObjectInspector(argOIs[0]);
                }

                processOptions(argOIs);
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

        @Nonnull
        private static StructObjectInspector internalMergeOI(@Nonnull ObjectInspector valueOI,
                @Nonnull PrimitiveObjectInspector keyOI) {
            List<String> fieldNames = new ArrayList<String>();
            List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

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
            if (parameters[0] == null) {
                return;
            }
            Object value = ObjectInspectorUtils.copyToStandardObject(parameters[0], valueOI);

            final Object key;
            if (sortByKey) {
                if (parameters[1] == null) {
                    return;
                }
                key = ObjectInspectorUtils.copyToStandardObject(parameters[1], keyOI);
            } else {
                // set value to key
                key = ObjectInspectorUtils.copyToStandardObject(parameters[0], valueOI);
            }

            TupleWithKey tuple = new TupleWithKey(key, value);
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;

            myagg.iterate(tuple);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;

            Pair<List<Object>, List<Object>> tuples = myagg.drainQueue();
            List<Object> keyList = tuples.getKey();
            List<Object> valueList = tuples.getValue();
            if (valueList.isEmpty()) {
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
        public List<Object> terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            QueueAggregationBuffer myagg = (QueueAggregationBuffer) agg;
            Pair<List<Object>, List<Object>> tuples = myagg.drainQueue();
            return tuples.getValue();
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

            void iterate(@Nonnull TupleWithKey tuple) {
                if (queueHandler == null) {
                    initQueueHandler();
                }
                queueHandler.offer(tuple);
            }

            void merge(@Nonnull List<Object> o_keyList, @Nonnull List<Object> o_valueList) {
                if (queueHandler == null) {
                    initQueueHandler();
                }
                for (int i = 0, n = o_keyList.size(); i < n; i++) {
                    queueHandler.offer(new TupleWithKey(o_keyList.get(i), o_valueList.get(i)));
                }
            }

            @Nonnull
            Pair<List<Object>, List<Object>> drainQueue() {
                int n = queueHandler.size();
                final Object[] keys = new Object[n];
                final Object[] values = new Object[n];
                for (int i = n - 1; i >= 0; i--) { // head element in queue should be stored to tail of array
                    TupleWithKey tuple = queueHandler.poll();
                    keys[i] = tuple.getKey();
                    values[i] = tuple.getValue();
                }
                queueHandler.clear();

                return Pair.of(Arrays.asList(keys), Arrays.asList(values));
            }

            private void initQueueHandler() {
                final Comparator<TupleWithKey> comparator;
                if (reverseOrder) {
                    comparator = Collections.reverseOrder();
                } else {
                    comparator = NaturalComparator.getInstance();
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

            abstract void offer(@Nonnull TupleWithKey tuple);

            abstract int size();

            @Nullable
            abstract TupleWithKey poll();

            abstract void clear();

        }

        private static final class QueueHandler extends AbstractQueueHandler {

            private static final int DEFAULT_INITIAL_CAPACITY = 11; // same as PriorityQueue

            @Nonnull
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

            @Nonnull
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
            @Nonnull
            private final Object key;
            @Nonnull
            private final Object value;

            TupleWithKey(@CheckForNull Object key, @CheckForNull Object value) {
                this.key = Preconditions.checkNotNull(key);
                this.value = Preconditions.checkNotNull(value);
            }

            @Nonnull
            Object getKey() {
                return key;
            }

            @Nonnull
            Object getValue() {
                return value;
            }

            @Override
            public int compareTo(TupleWithKey o) {
                @SuppressWarnings("unchecked")
                Comparable<? super Object> k = (Comparable<? super Object>) key;
                return k.compareTo(o.getKey());
            }
        }
    }
}
