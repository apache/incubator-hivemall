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
package hivemall.fm;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.SizeOf;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AggregationType;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

@Description(name = "ffm_predict",
        value = "_FUNC_(float Wi, array<float> Vifj, array<float> Vjfi, float Xi, float Xj)"
                + " - Returns a prediction value in Double")
public final class FFMPredictGenericUDAF extends AbstractGenericUDAFResolver {

    private FFMPredictGenericUDAF() {}

    @Override
    public Evaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 5) {
            throw new UDFArgumentLengthException(
                "Expected argument length is 5 but given argument length was " + typeInfo.length);
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[0])) {
            throw new UDFArgumentTypeException(0,
                "Number type is expected for the first argument Wi: " + typeInfo[0].getTypeName());
        }
        if (typeInfo[1].getCategory() != Category.LIST) {
            throw new UDFArgumentTypeException(1,
                "List type is expected for the second argument Vifj: " + typeInfo[1].getTypeName());
        }
        if (typeInfo[2].getCategory() != Category.LIST) {
            throw new UDFArgumentTypeException(2,
                "List type is expected for the third argument Vjfi: " + typeInfo[2].getTypeName());
        }
        ListTypeInfo typeInfo1 = (ListTypeInfo) typeInfo[1];
        if (!HiveUtils.isFloatingPointTypeInfo(typeInfo1.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(1,
                "Double or Float type is expected for the element type of list Vifj: "
                        + typeInfo1.getTypeName());
        }
        ListTypeInfo typeInfo2 = (ListTypeInfo) typeInfo[2];
        if (!HiveUtils.isFloatingPointTypeInfo(typeInfo2.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(2,
                "Double or Float type is expected for the element type of list Vjfi: "
                        + typeInfo1.getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[3])) {
            throw new UDFArgumentTypeException(3,
                "Number type is expected for the third argument Xi: " + typeInfo[3].getTypeName());
        }
        if (!HiveUtils.isNumberTypeInfo(typeInfo[4])) {
            throw new UDFArgumentTypeException(4,
                "Number type is expected for the third argument Xi: " + typeInfo[4].getTypeName());
        }
        return new Evaluator();
    }

    public static final class Evaluator extends GenericUDAFEvaluator {

        // input OI
        private PrimitiveObjectInspector wiOI;
        private ListObjectInspector vijOI, vjiOI;
        private PrimitiveObjectInspector vijElemOI, vjiElemOI;
        private PrimitiveObjectInspector xiOI, xjOI;

        // merge input OI
        private DoubleObjectInspector mergeInputOI;

        public Evaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 5);
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.wiOI = HiveUtils.asDoubleCompatibleOI(parameters[0]);
                this.vijOI = HiveUtils.asListOI(parameters[1]);
                this.vijElemOI = HiveUtils.asFloatingPointOI(vijOI.getListElementObjectInspector());
                this.vjiOI = HiveUtils.asListOI(parameters[2]);
                this.vjiElemOI = HiveUtils.asFloatingPointOI(vjiOI.getListElementObjectInspector());
                this.xiOI = HiveUtils.asDoubleCompatibleOI(parameters[3]);
                this.xjOI = HiveUtils.asDoubleCompatibleOI(parameters[4]);
            } else {// from partial aggregation
                this.mergeInputOI = HiveUtils.asDoubleOI(parameters[0]);
            }

            return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
        }

        @Override
        public FFMPredictAggregationBuffer getNewAggregationBuffer() throws HiveException {
            FFMPredictAggregationBuffer myAggr = new FFMPredictAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;
            myAggr.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            final FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;

            if (parameters[0] == null) {//  Wi is null
                if (parameters[3] == null || parameters[4] == null) {
                    // both Xi and Xj are nonnull => <Vifj, Vjfi> Xi Xj
                    return;
                }
                if (parameters[1] == null || parameters[2] == null) {
                    // vi, vj can be null where feature index does not exist in the prediction model  
                    return;
                }

                // (i, j, xi, xj) => (wi, vi, vj, xi, xj)
                float[] vij = HiveUtils.asFloatArray(parameters[1], vijOI, vijElemOI, false);
                float[] vji = HiveUtils.asFloatArray(parameters[2], vjiOI, vjiElemOI, false);
                double xi = PrimitiveObjectInspectorUtils.getDouble(parameters[3], xiOI);
                double xj = PrimitiveObjectInspectorUtils.getDouble(parameters[4], xjOI);

                myAggr.addViVjXiXj(vij, vji, xi, xj);
            } else {
                final double wi = PrimitiveObjectInspectorUtils.getDouble(parameters[0], wiOI);

                if (parameters[3] == null && parameters[4] == null) {// Xi and Xj are null => global bias `w0`
                    // (i=0, j=null, xi=null, xj=null) => (wi, vi=?, vj=null, xi=null, xj=null)
                    myAggr.addW0(wi);
                } else if (parameters[4] == null) {// Only Xi is nonnull => linear combination `wi` * `xi`
                    // (i, j=null, xi, xj=null) => (wi, vi, vj=null, xi, xj=null)
                    double xi = PrimitiveObjectInspectorUtils.getDouble(parameters[3], xiOI);
                    myAggr.addWiXi(wi, xi);
                }
            }
        }

        @Override
        public DoubleWritable terminatePartial(
                @SuppressWarnings("deprecation") AggregationBuffer agg) throws HiveException {
            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;
            double sum = myAggr.get();
            return new DoubleWritable(sum);
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;
            double sum = mergeInputOI.get(partial);
            myAggr.merge(sum);
        }

        @Override
        public DoubleWritable terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }

    }

    @AggregationType(estimable = true)
    public static final class FFMPredictAggregationBuffer extends AbstractAggregationBuffer {

        private double sum;

        FFMPredictAggregationBuffer() {
            super();
        }

        void reset() {
            this.sum = 0.d;
        }

        void merge(double o_sum) {
            this.sum += o_sum;
        }

        double get() {
            return sum;
        }

        void addW0(final double W0) {
            this.sum += W0;
        }

        void addWiXi(final double Wi, final double Xi) {
            this.sum += (Wi * Xi);
        }

        void addViVjXiXj(@Nonnull final float[] Vij, @Nonnull final float[] Vji, final double Xi,
                final double Xj) throws UDFArgumentException {
            if (Vij.length != Vji.length) {
                throw new UDFArgumentException("Mismatch in the number of factors");
            }

            final int factors = Vij.length;

            // compute inner product <Vifj, Vjfi>
            double prod = 0.d;
            for (int f = 0; f < factors; f++) {
                prod += (Vij[f] * Vji[f]);
            }

            this.sum += (prod * Xi * Xj);
        }

        @Override
        public int estimate() {
            return SizeOf.DOUBLE;
        }

    }

}
