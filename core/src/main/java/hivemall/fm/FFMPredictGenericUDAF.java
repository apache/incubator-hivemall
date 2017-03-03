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
import hivemall.utils.hadoop.WritableUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.WritableDoubleObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;

@Description(
        name = "ffm_predict",
        value = "_FUNC_(Float Wi, Float Wj, array<float> Vifj, array<float> Vjfi, float Xi, float Xj)"
                + " - Returns a prediction value in Double")
public final class FFMPredictGenericUDAF extends AbstractGenericUDAFResolver {

    private FFMPredictGenericUDAF() {}

    @Override
    public Evaluator getEvaluator(TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 5) {
            throw new UDFArgumentLengthException(
                "Expected argument length is 6 but given argument length was " + typeInfo.length);
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
        if (!HiveUtils.isNumberTypeInfo(typeInfo1.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(1,
                "Number type is expected for the element type of list Vifj: "
                        + typeInfo1.getTypeName());
        }
        ListTypeInfo typeInfo2 = (ListTypeInfo) typeInfo[2];
        if (!HiveUtils.isNumberTypeInfo(typeInfo2.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(2,
                "Number type is expected for the element type of list Vjfi: "
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

    public static class Evaluator extends GenericUDAFEvaluator {

        // input OI
        private PrimitiveObjectInspector wiOI;
        private ListObjectInspector vijOI;
        private ListObjectInspector vjiOI;
        private PrimitiveObjectInspector xiOI;
        private PrimitiveObjectInspector xjOI;

        // merge OI
        private StructObjectInspector internalMergeOI;
        private StructField sumField;

        public Evaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 5);
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.wiOI = HiveUtils.asDoubleCompatibleOI(parameters[0]);
                this.vijOI = HiveUtils.asListOI(parameters[1]);
                this.vjiOI = HiveUtils.asListOI(parameters[2]);
                this.xiOI = HiveUtils.asDoubleCompatibleOI(parameters[3]);
                this.xjOI = HiveUtils.asDoubleCompatibleOI(parameters[4]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.sumField = soi.getStructFieldRef("sum");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {
                outputOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("sum");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
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
            if (parameters[0] == null) {
                return;
            }
            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;

            double wi = PrimitiveObjectInspectorUtils.getDouble(parameters[0], wiOI);
            if (parameters[3] == null && parameters[4] == null) {// Xi and Xj are null => global bias `w0`
                myAggr.addW0(wi);
            } else if (parameters[4] == null) {// Only Xi is nonnull => linear combination `wi` * `xi`
                double xi = PrimitiveObjectInspectorUtils.getDouble(parameters[3], xiOI);
                myAggr.addWiXi(wi, xi);
            } else {// both Xi and Xj are nonnull => <Vifj, Vjfi> Xi Xj
                if (parameters[1] == null || parameters[2] == null) {
                    throw new UDFArgumentException("The second and third arguments (Vij, Vji) must not be null");
                }

                List<Float> vij = (List<Float>) vijOI.getList(parameters[1]);
                List<Float> vji = (List<Float>) vjiOI.getList(parameters[2]);

                if (vij.size() != vji.size()) {
                    throw new HiveException("Mismatch in the number of factors");
                }

                double xi = PrimitiveObjectInspectorUtils.getDouble(parameters[3], xiOI);
                double xj = PrimitiveObjectInspectorUtils.getDouble(parameters[4], xjOI);

                myAggr.addViVjXiXj(vij, vji, xi, xj);
            }
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;

            final Object[] partialResult = new Object[1];
            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            Object sumObj = internalMergeOI.getStructFieldData(partial, sumField);
            double sum = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(sumObj);

            FFMPredictAggregationBuffer myAggr = (FFMPredictAggregationBuffer) agg;
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

    public static class FFMPredictAggregationBuffer extends AbstractAggregationBuffer {

        double sum;

        FFMPredictAggregationBuffer() {
            super();
        }

        void reset() {
            this.sum = 0.d;
        }

        void merge(final double o_sum) {
            sum += o_sum;
        }

        double get() {
            return sum;
        }

        void addW0(final double W0) {
            sum += W0;
        }

        void addWiXi(final double Wi, final double Xi) {
            sum += Wi * Xi;
        }

        void addViVjXiXj(@Nonnull final List<Float> Vifj, @Nonnull final List<Float> Vjfi,
                     final double Xi, final double Xj) {
            final int factors = Vifj.size();
            double prod = 0.d;

            // compute inner product <Vifj, Vjfi>
            for (int i = 0; i < factors; i++) {
                prod += (Vifj.get(i) * Vjfi.get(i));
            }

            sum += (prod * Xi * Xj);
        }

    }

}
