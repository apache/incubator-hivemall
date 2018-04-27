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
package hivemall.sketch.hll;

import hivemall.UDAFEvaluatorWithOptions;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;

import java.io.IOException;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AggregationType;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BinaryObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.LongWritable;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

@Description(name = "approx_count_distinct", value = "_FUNC_(expr x [, const string options])"
        + " - Returns an approximation of count(DISTINCT x) using HyperLogLogPlus algorithm")
public final class ApproxCountDistinctUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo)
            throws SemanticException {
        if (typeInfo.length != 1 && typeInfo.length != 2) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "_FUNC_ takes one or two arguments");
        }
        if (typeInfo.length == 2 && !HiveUtils.isStringTypeInfo(typeInfo[1])) {
            throw new UDFArgumentTypeException(1,
                "The second argument type expected to be const string: " + typeInfo[1]);
        }

        return new HLLEvaluator();
    }

    public static final class HLLEvaluator extends UDAFEvaluatorWithOptions {

        @Nullable
        private int[] params;

        private ObjectInspector origInputOI;
        private BinaryObjectInspector mergeInputOI;

        @Override
        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("p", true,
                "The size of registers for the normal set. `p` MUST be in the range [4,sp] and 15 by the default");
            opts.addOption("sp", true,
                "The size of registers for the sparse set. `sp` MUST be in the range [4,32] and 25 by the default");
            return opts;
        }

        @Override
        protected CommandLine processOptions(@Nonnull ObjectInspector[] argOIs)
                throws UDFArgumentException {
            CommandLine cl = null;

            int p = 15, sp = 25;
            if (argOIs.length == 2) {
                if (!HiveUtils.isConstString(argOIs[1])) {
                    throw new UDFArgumentException(
                        "The second argument type expected to be const string: " + argOIs[1]);
                }
                cl = parseOptions(HiveUtils.getConstString(argOIs[1]));

                p = Primitives.parseInt(cl.getOptionValue("p"), p);
                sp = Primitives.parseInt(cl.getOptionValue("sp"), sp);
                validateArguments(p, sp);
            }

            this.params = new int[] {p, sp};

            return cl;
        }

        @Override
        public ObjectInspector init(@Nonnull Mode mode, @Nonnull ObjectInspector[] parameters)
                throws HiveException {
            assert (parameters.length == 1 || parameters.length == 2) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                processOptions(parameters);
                this.origInputOI = parameters[0];
            } else {// from partial aggregation
                this.mergeInputOI = HiveUtils.asBinaryOI(parameters[0]);
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = PrimitiveObjectInspectorFactory.javaByteArrayObjectInspector;
            } else {// terminate
                outputOI = PrimitiveObjectInspectorFactory.writableLongObjectInspector;
            }
            return outputOI;
        }

        @Override
        public HLLBuffer getNewAggregationBuffer() throws HiveException {
            HLLBuffer buf = new HLLBuffer();
            if (params != null) {
                buf.reset(params[0], params[1]);
            }
            return buf;
        }

        @SuppressWarnings("deprecation")
        @Override
        public void reset(@Nonnull AggregationBuffer agg) throws HiveException {
            HLLBuffer buf = (HLLBuffer) agg;
            if (params != null) {
                buf.reset(params[0], params[1]);
            } else {
                buf.hll = null;
            }
        }

        @SuppressWarnings("deprecation")
        @Override
        public void iterate(@Nonnull AggregationBuffer agg, @Nonnull Object[] parameters)
                throws HiveException {
            if (parameters[0] == null) {
                return;
            }

            HLLBuffer buf = (HLLBuffer) agg;
            Object value =
                    ObjectInspectorUtils.copyToStandardJavaObject(parameters[0], origInputOI);
            Preconditions.checkNotNull(buf.hll, HiveException.class);
            buf.hll.offer(value);
        }

        @SuppressWarnings("deprecation")
        @Override
        @Nullable
        public byte[] terminatePartial(@Nonnull AggregationBuffer agg) throws HiveException {
            HLLBuffer buf = (HLLBuffer) agg;
            if (buf.hll == null) {
                return null;
            }
            try {
                return buf.hll.getBytes();
            } catch (IOException e) {
                throw new HiveException(e);
            }
        }

        @SuppressWarnings("deprecation")
        @Override
        public void merge(@Nonnull AggregationBuffer agg, @Nullable Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            byte[] data = mergeInputOI.getPrimitiveJavaObject(partial);
            final HyperLogLogPlus otherHLL;
            try {
                otherHLL = HyperLogLogPlus.Builder.build(data);
            } catch (IOException e) {
                throw new HiveException("Failed to build other HLL");
            }

            final HLLBuffer buf = (HLLBuffer) agg;
            if (buf.hll == null) {
                buf.hll = otherHLL;
            } else {
                try {
                    buf.hll.addAll(otherHLL);
                } catch (CardinalityMergeException e) {
                    throw new HiveException("Failed to merge HLL");
                }
            }
        }

        @SuppressWarnings("deprecation")
        @Override
        public LongWritable terminate(@Nonnull AggregationBuffer agg) throws HiveException {
            HLLBuffer buf = (HLLBuffer) agg;

            long cardinality = (buf.hll == null) ? 0L : buf.hll.cardinality();
            return new LongWritable(cardinality);
        }

    }

    private static void validateArguments(final int p, final int sp) throws UDFArgumentException {
        if (p < 4 || p > sp) {
            throw new UDFArgumentException("p must be between 4 and sp (inclusive)");
        }
        if (sp > 32) {
            throw new UDFArgumentException("sp values greater than 32 not supported");
        }
    }

    @AggregationType(estimable = true)
    static final class HLLBuffer extends AbstractAggregationBuffer {

        @Nullable
        private HyperLogLogPlus hll;

        HLLBuffer() {}

        @Override
        public int estimate() {
            return (hll == null) ? 0 : hll.sizeof();
        }

        void reset(@Nonnegative int p, @Nonnegative int sp) {
            this.hll = new HyperLogLogPlus(p, sp);
        }

    }
}
