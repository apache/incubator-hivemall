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
package hivemall.tools.aggr;

import static hivemall.utils.hadoop.HiveUtils.asLongOI;
import static hivemall.utils.hadoop.HiveUtils.asPrimitiveObjectInspector;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.LongCounter;
import hivemall.utils.lang.Preconditions;

import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AggregationType;
import org.apache.hadoop.hive.ql.util.JavaDataModel;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardMapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

//@formatter:off
@Description(name = "majority_vote",
        value = "_FUNC_(Primitive x) - Returns the most frequent value of x",
        extended = "-- see https://issues.apache.org/jira/browse/HIVE-17406 \n" +
                   "WITH data as (\n" + 
                   "  select\n" + 
                   "    explode(array('1', '2', '2', '2', '5', '4', '1', '2')) as k\n" + 
                   ")\n" + 
                   "select\n" + 
                   "  majority_vote(k) as k\n" + 
                   "from \n" + 
                   "  data;\n" + 
                   "2")
//@formatter:on
public final class MajorityVoteUDAF extends AbstractGenericUDAFResolver {

    public MajorityVoteUDAF() {
        super();
    }

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull final TypeInfo[] argTypes)
            throws SemanticException {
        if (argTypes.length != 1) {
            throw new UDFArgumentLengthException(
                "Expected ecactly one argument: " + argTypes.length);
        }
        if (!HiveUtils.isPrimitiveTypeInfo(argTypes[0])) {
            throw new UDFArgumentTypeException(0,
                "PRIMITIVE type is expected but got " + argTypes[0]);
        }
        return new Evaluator();
    }

    public static final class Evaluator extends GenericUDAFEvaluator {

        // original input
        private transient PrimitiveObjectInspector keyInputOI;
        private transient PrimitiveObjectInspector keyOutputOI;

        // partial aggregation
        private transient StandardMapObjectInspector partialOI;
        private transient LongObjectInspector counterInputOI;

        public Evaluator() {
            super();
        }

        @Override
        public ObjectInspector init(@Nonnull Mode mode, @Nonnull ObjectInspector[] argOIs)
                throws HiveException {
            assert (argOIs.length == 1);
            super.init(mode, argOIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.keyInputOI = asPrimitiveObjectInspector(argOIs[0]);
            } else {// from partial aggregation
                this.partialOI = (StandardMapObjectInspector) argOIs[0];
                this.keyInputOI = asPrimitiveObjectInspector(partialOI.getMapKeyObjectInspector());
                this.counterInputOI = asLongOI(partialOI.getMapValueObjectInspector());
            }
            this.keyOutputOI = PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(
                keyInputOI.getTypeInfo());

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = ObjectInspectorFactory.getStandardMapObjectInspector(keyOutputOI,
                    PrimitiveObjectInspectorFactory.javaLongObjectInspector);
            } else {// terminate
                outputOI = keyOutputOI;
            }
            return outputOI;
        }

        @Override
        public CounterBuffer getNewAggregationBuffer() throws HiveException {
            CounterBuffer buf = new CounterBuffer();
            buf.reset();
            return buf;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            CounterBuffer buf = (CounterBuffer) agg;
            buf.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            CounterBuffer buf = (CounterBuffer) agg;

            final Object param0 = parameters[0];
            if (param0 == null) {
                return;
            }

            final Object key = keyInputOI.getPrimitiveJavaObject(param0);
            if (key != null) {
                buf.iterate(key);
            }
        }

        @Override
        public Map<Object, Long> terminatePartial(
                @SuppressWarnings("deprecation") AggregationBuffer agg) throws HiveException {
            CounterBuffer buf = (CounterBuffer) agg;

            return buf.terminatePartial();
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            final CounterBuffer buf = (CounterBuffer) agg;
            Map<?, ?> partialResult = partialOI.getMap(partial);
            for (Map.Entry<?, ?> e : partialResult.entrySet()) {
                Object k =
                        keyInputOI.getPrimitiveJavaObject(Preconditions.checkNotNull(e.getKey()));
                long v = counterInputOI.get(Preconditions.checkNotNull(e.getValue()));
                buf.merge(k, v);
            }
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            CounterBuffer buf = (CounterBuffer) agg;
            return buf.terminate();
        }

    }

    @AggregationType(estimable = true)
    public static final class CounterBuffer extends AbstractAggregationBuffer {

        @Nonnull
        private LongCounter<Object> partial;

        public CounterBuffer() {
            super();
            reset();
        }

        void reset() {
            this.partial = new LongCounter<Object>();
        }

        void iterate(@Nonnull final Object k) {
            partial.increment(k);
        }

        void merge(@Nonnull final Object k, final long v) {
            partial.increment(k, v);
        }

        @Nonnull
        Map<Object, Long> terminatePartial() {
            return partial.getMap();
        }

        @Nullable
        Object terminate() {
            return partial.whichMax();
        }

        @Override
        public int estimate() {
            int size = partial.size();
            return JavaDataModel.PRIMITIVES2 * size * 2; // rough estimate
        }

    }

}
