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
package hivemall.classifier;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AggregationType;
import org.apache.hadoop.hive.ql.util.JavaDataModel;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

@Description(name = "kpa_predict",
        value = "_FUNC_(@Nonnull double xh, @Nonnull double xk, @Nullable float w0, @Nonnull float w1, @Nonnull float w2, @Nullable float w3)"
                + " - Returns a prediction value in Double")
public final class KPAPredictUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(TypeInfo[] parameters) throws SemanticException {
        if (parameters.length != 6) {
            throw new UDFArgumentException(
                "_FUNC_(double xh, double xk, float w0, float w1, float w2, float w3) takes exactly 6 arguments but got: "
                        + parameters.length);
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[0])) {
            throw new UDFArgumentTypeException(0,
                "Number type is expected for xh (1st argument): " + parameters[0].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[1])) {
            throw new UDFArgumentTypeException(1,
                "Number type is expected for xk (2nd argument): " + parameters[1].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[2])) {
            throw new UDFArgumentTypeException(2,
                "Number type is expected for w0 (3rd argument): " + parameters[2].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[3])) {
            throw new UDFArgumentTypeException(3,
                "Number type is expected for w1 (4th argument): " + parameters[3].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[4])) {
            throw new UDFArgumentTypeException(4,
                "Number type is expected for w2 (5th argument): " + parameters[4].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(parameters[5])) {
            throw new UDFArgumentTypeException(5,
                "Number type is expected for w3 (6th argument): " + parameters[5].getTypeName());
        }

        return new Evaluator();
    }

    public static class Evaluator extends GenericUDAFEvaluator {

        @Nullable
        private transient PrimitiveObjectInspector xhOI, xkOI;
        @Nullable
        private transient PrimitiveObjectInspector w0OI, w1OI, w2OI, w3OI;

        public Evaluator() {}

        @Override
        public ObjectInspector init(Mode m, ObjectInspector[] parameters) throws HiveException {
            super.init(m, parameters);

            // initialize input
            if (m == Mode.PARTIAL1 || m == Mode.COMPLETE) {// from original data
                this.xhOI = HiveUtils.asNumberOI(parameters[0]);
                this.xkOI = HiveUtils.asNumberOI(parameters[1]);
                this.w0OI = HiveUtils.asNumberOI(parameters[2]);
                this.w1OI = HiveUtils.asNumberOI(parameters[3]);
                this.w2OI = HiveUtils.asNumberOI(parameters[4]);
                this.w3OI = HiveUtils.asNumberOI(parameters[5]);
            }

            return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
        }

        @Override
        public AggrBuffer getNewAggregationBuffer() throws HiveException {
            return new AggrBuffer();
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            AggrBuffer aggr = (AggrBuffer) agg;
            aggr.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            Preconditions.checkArgument(parameters.length == 6, HiveException.class);

            final AggrBuffer aggr = (AggrBuffer) agg;

            if (parameters[0] /* xh */ != null) {
                double xh = HiveUtils.getDouble(parameters[0], xhOI);
                if (parameters[1] /* xk */ != null) {
                    if (parameters[5] /* w3hk */ == null) {
                        return;
                    }
                    // xh, xk, w3hk
                    double xk = HiveUtils.getDouble(parameters[1], xkOI);
                    double w3hk = HiveUtils.getDouble(parameters[5], w3OI);
                    aggr.addW3(xh, xk, w3hk);
                } else {
                    if (parameters[3] /* w1h */ == null) {
                        return;
                    }
                    // xh, w1h, w2h
                    Preconditions.checkNotNull(parameters[4], HiveException.class);
                    double w1h = HiveUtils.getDouble(parameters[3], w1OI);
                    double w2h = HiveUtils.getDouble(parameters[4], w2OI);
                    aggr.addW1W2(xh, w1h, w2h);
                }
            } else if (parameters[2] /* w0 */ != null) {
                // w0
                double w0 = HiveUtils.getDouble(parameters[2], w0OI);
                aggr.addW0(w0);
            }
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            AggrBuffer aggr = (AggrBuffer) agg;
            double v = aggr.get();
            return new DoubleWritable(v);
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            AggrBuffer aggr = (AggrBuffer) agg;
            DoubleWritable other = (DoubleWritable) partial;
            double v = other.get();
            aggr.merge(v);
        }

        @Override
        public DoubleWritable terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            AggrBuffer aggr = (AggrBuffer) agg;
            double v = aggr.get();
            return new DoubleWritable(v);
        }

    }

    @AggregationType(estimable = true)
    static class AggrBuffer extends AbstractAggregationBuffer {

        double score;

        AggrBuffer() {
            super();
            reset();
        }

        @Override
        public int estimate() {
            return JavaDataModel.PRIMITIVES2;
        }

        void reset() {
            this.score = 0.d;
        }

        double get() {
            return score;
        }

        void addW0(@Nonnull double w0) {
            this.score += w0;
        }

        void addW1W2(final double xh, final double w1h, final double w2h) {
            this.score += (w1h * xh + w2h * xh * xh);
        }

        void addW3(final double xh, final double xk, final double w3hk) {
            this.score += (w3hk * xh * xk);
        }

        void merge(final double other) {
            this.score += other;
        }

    }

}
