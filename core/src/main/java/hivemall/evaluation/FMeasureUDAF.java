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
package hivemall.evaluation;

import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.*;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.LongWritable;

import javax.annotation.Nonnull;

@Description(name = "fmeasure",
        value = "_FUNC_(array[int], array[int], double) - Return a F-measure score")
public final class FMeasureUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 2 && typeInfo.length != 3) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                    "_FUNC_ takes two or three arguments");
        }

        ListTypeInfo arg1type = HiveUtils.asListTypeInfo(typeInfo[0]);
        if (!HiveUtils.isPrimitiveTypeInfo(arg1type.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(0,
                    "The first argument `array actual` is invalid form: " + typeInfo[0]);
        }
        ListTypeInfo arg2type = HiveUtils.asListTypeInfo(typeInfo[1]);
        if (!HiveUtils.isPrimitiveTypeInfo(arg2type.getListElementTypeInfo())) {
            throw new UDFArgumentTypeException(1,
                    "The second argument `array predicted` is invalid form: " + typeInfo[1]);
        }

        return new Evaluator();
    }

    public static class Evaluator extends GenericUDAFEvaluator {

        private ListObjectInspector actualOI;
        private ListObjectInspector predictedOI;
        private PrimitiveObjectInspector betaOI;
        private StructObjectInspector internalMergeOI;

        private StructField tpField;
        private StructField totalActualField;
        private StructField totalPredictedField;
        private StructField betaField;


        public Evaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 2 || parameters.length == 3) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.actualOI = (ListObjectInspector) parameters[0];
                this.predictedOI = (ListObjectInspector) parameters[1];
                if (parameters.length == 3) {
                    this.betaOI = HiveUtils.asNumberOI(parameters[2]);
                }
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.tpField = soi.getStructFieldRef("tp");
                this.totalActualField = soi.getStructFieldRef("totalActual");
                this.totalPredictedField = soi.getStructFieldRef("totalPredicted");
                this.betaField = soi.getStructFieldRef("beta");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {// terminate
                outputOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<>();

            fieldNames.add("tp");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("totalActual");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("totalPredicted");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("beta");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @Override
        public FMeasureAggregationBuffer getNewAggregationBuffer() throws HiveException {
            FMeasureAggregationBuffer myAggr = new FMeasureAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            myAggr.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                            Object[] parameters) throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;

            List<?> actual = actualOI.getList(parameters[0]);
            if (actual == null) {
                actual = Collections.emptyList();
            }

            List<?> predicted = predictedOI.getList(parameters[1]);
            if (predicted == null) {
                return;
            }

            double beta = 1.d;
            if (parameters.length == 3) {
                beta = HiveUtils.getDouble(parameters[2], betaOI);
            }
            if (beta <= 0.d) {
                throw new UDFArgumentException(
                        "The third argument `double beta` must be greater than 0.0");
            }

            myAggr.iterate(actual, predicted, beta);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;

            Object[] partialResult = new Object[4];
            partialResult[0] = new LongWritable(myAggr.tp);
            partialResult[1] = new LongWritable(myAggr.totalActual);
            partialResult[2] = new LongWritable(myAggr.totalPredicted);
            partialResult[3] = new DoubleWritable(myAggr.beta);
            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            Object tpObj = internalMergeOI.getStructFieldData(partial, tpField);
            Object totalActualObj = internalMergeOI.getStructFieldData(partial, totalActualField);
            Object totalPredictedObj = internalMergeOI.getStructFieldData(partial, totalPredictedField);
            Object betaObj = internalMergeOI.getStructFieldData(partial, betaField);
            long tp = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(tpObj);
            long totalActual = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(totalActualObj);
            long totalPredicted = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(totalPredictedObj);
            double beta = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(betaObj);

            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            myAggr.merge(tp, totalActual, totalPredicted, beta);
        }

        @Override
        public DoubleWritable terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }
    }

    public static class FMeasureAggregationBuffer extends GenericUDAFEvaluator.AbstractAggregationBuffer {
        long tp;
        /** tp + fn */
        long totalActual;
        /** tp + fp */
        long totalPredicted;
        double beta;

        public FMeasureAggregationBuffer() { super(); }

        void reset() {
            this.tp = 0L;
            this.totalActual = 0L;
            this.totalPredicted = 0L;
        }


        void merge(long o_tp, long o_actual, long o_predicted, double beta) {
            tp += o_tp;
            totalActual += o_actual;
            totalPredicted += o_predicted;
            this.beta = beta;
        }

        double get() {
            double precision = precision(tp, totalPredicted);
            double recall = recall(tp, totalActual);
            double squareBeta = Math.pow(beta, 2.d);

            double divisor = squareBeta * precision + recall;
            if (divisor > 0) {
                return ((1.d + squareBeta) * precision * recall) / divisor;
            } else {
                return -1d;
            }
        }

        private static double precision(long tp, long totalPredicted) {
            return (totalPredicted == 0L) ? 0d : tp
                    / (double) totalPredicted;
        }

        private static double recall(long tp, long totalActual) {
            return (totalActual == 0L) ? 0d : tp / (double) totalActual;
        }

        void iterate(@Nonnull List<?> actual, @Nonnull List<?> predicted, @Nonnull double beta) {
            final int numActual = actual.size();
            final int numPredicted = predicted.size();
            int countTp = 0;

            for (int i = 0; i < numPredicted; i++) {
                Object p = predicted.get(i);
                if (actual.contains(p)) {
                    countTp++;
                }
            }
            this.tp += countTp;
            this.totalActual += numActual;
            this.totalPredicted += numPredicted;
            this.beta = beta;
        }
    }
}
