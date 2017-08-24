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

import hivemall.UDAFEvaluatorWithOptions;
import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.LongWritable;

import javax.annotation.Nonnull;

@Description(
        name = "fmeasure",
        value = "_FUNC_(array | int | boolean actual , array | int | boolean predicted, String) - Return a F-measure (f1score is the special with beta=1.)")
public final class FMeasureUDAF extends AbstractGenericUDAFResolver {
    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 2 && typeInfo.length != 3) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "_FUNC_ takes two or three arguments");
        }

        boolean isArg1ListOrIntOrBoolean = HiveUtils.isListTypeInfo(typeInfo[0])
                || HiveUtils.isIntegerTypeInfo(typeInfo[0])
                || HiveUtils.isBooleanTypeInfo(typeInfo[0]);
        if (!isArg1ListOrIntOrBoolean) {
            throw new UDFArgumentTypeException(0,
                "The first argument `array/int/boolean actual` is invalid form: " + typeInfo[0]);
        }

        boolean isArg2ListOrIntOrBoolean = HiveUtils.isListTypeInfo(typeInfo[1])
                || HiveUtils.isIntegerTypeInfo(typeInfo[1])
                || HiveUtils.isBooleanTypeInfo(typeInfo[1]);
        if (!isArg2ListOrIntOrBoolean) {
            throw new UDFArgumentTypeException(1,
                "The second argument `array/int/boolean actual` is invalid form: " + typeInfo[1]);
        }

        if (typeInfo[0] != typeInfo[1]) {
            throw new UDFArgumentTypeException(1, "The first argument's `actual` type is "
                    + typeInfo[0] + ", but the second argument `predicated`'s type is not match: "
                    + typeInfo[1]);
        }

        return new Evaluator();
    }

    public static class Evaluator extends UDAFEvaluatorWithOptions {

        private ObjectInspector actualOI;
        private ObjectInspector predictedOI;
        private StructObjectInspector internalMergeOI;

        private StructField tpField;
        private StructField totalActualField;
        private StructField totalPredictedField;
        private StructField betaOptionField;
        private StructField averageOptionFiled;

        private double beta;
        private String average;

        public Evaluator() {}

        @Override
        protected Options getOptions() {
            Options opts = new Options();
            opts.addOption("beta", true, "The weight of precision [default: 1.]");
            opts.addOption("average", true, "The way of average calculation [default: micro]");
            return opts;
        }

        @Override
        protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
            CommandLine cl = null;

            double beta = 1.0d;
            String average = "micro";

            if (argOIs.length >= 3) {
                String rawArgs = HiveUtils.getConstString(argOIs[2]);
                cl = parseOptions(rawArgs);

                beta = Primitives.parseDouble(cl.getOptionValue("beta"), beta);
                if (beta <= 0.d) {
                    throw new UDFArgumentException(
                        "The third argument `double beta` must be greater than 0.0: " + beta);
                }

                average = cl.getOptionValue("average", "micro");

                if (average.equals("macro")) {
                    throw new UDFArgumentException("\"-average macro\" is not supported");
                }

                if (!(average.equals("binary") || average.equals("micro"))) {
                    throw new UDFArgumentException(
                        "The third argument `String average` must be one of the {binary, micro, macro}: "
                                + average);
                }
            }

            this.beta = beta;
            this.average = average;
            return cl;
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 2 || parameters.length == 3) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.processOptions(parameters);
                this.actualOI = parameters[0];
                this.predictedOI = parameters[1];
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.tpField = soi.getStructFieldRef("tp");
                this.totalActualField = soi.getStructFieldRef("totalActual");
                this.totalPredictedField = soi.getStructFieldRef("totalPredicted");
                this.betaOptionField = soi.getStructFieldRef("beta");
                this.averageOptionFiled = soi.getStructFieldRef("average");
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
            fieldNames.add("average");
            fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);

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
            myAggr.setOptions(this.beta, this.average);
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            boolean isList = HiveUtils.isListOI(actualOI) && HiveUtils.isListOI(predictedOI);

            List<?> actual = Collections.emptyList();
            List<?> predicted = Collections.emptyList();

            if (isList) {// array case
                if (this.average.equals("binary")) {
                    throw new UDFArgumentException(
                        "\"-average binary\" is not supported when `predict` is array");
                }
                actual = ((ListObjectInspector) actualOI).getList(parameters[0]);
                predicted = ((ListObjectInspector) predictedOI).getList(parameters[1]);
            } else {//binary case
                if (HiveUtils.isBooleanOI(actualOI)) { // boolean case
                    if (((BooleanObjectInspector) actualOI).get(parameters[0])) {
                        actual = Arrays.asList(1);
                    } else {
                        actual = Arrays.asList(0);
                    }

                    if (((BooleanObjectInspector) predictedOI).get(parameters[1])) {
                        predicted = Arrays.asList(1);
                    } else {
                        predicted = Arrays.asList(0);
                    }
                } else { // int case
                    int actualOIValue = ((IntObjectInspector) actualOI).get(parameters[0]);
                    if (actualOIValue == 1) {
                        actual = Arrays.asList(1);
                    } else if (!(actualOIValue == 0 || actualOIValue == -1)) {
                        throw new UDFArgumentException(
                            "The first argument `int actual` must be 1, 0, or -1:" + actualOIValue);
                    } else if (!this.average.equals("binary")) {
                        actual = Arrays.asList(0);
                    }

                    int predictedOIValue = ((IntObjectInspector) predictedOI).get(parameters[1]);
                    if (predictedOIValue == 1) {
                        predicted = Arrays.asList(1);
                    } else if (!(predictedOIValue == 0 || predictedOIValue == -1)) {
                        throw new UDFArgumentException(
                            "The second argument `int predicted` must be 1, 0, or -1:"
                                    + predictedOIValue);
                    } else if (!this.average.equals("binary")) {
                        predicted = Arrays.asList(0);
                    }
                }
            }
            myAggr.iterate(actual, predicted);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;

            Object[] partialResult = new Object[5];
            partialResult[0] = new LongWritable(myAggr.tp);
            partialResult[1] = new LongWritable(myAggr.totalActual);
            partialResult[2] = new LongWritable(myAggr.totalPredicted);
            partialResult[3] = new DoubleWritable(myAggr.beta);
            partialResult[4] = myAggr.average;
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
            Object totalPredictedObj = internalMergeOI.getStructFieldData(partial,
                totalPredictedField);
            Object betaObj = internalMergeOI.getStructFieldData(partial, betaOptionField);
            Object averageObj = internalMergeOI.getStructFieldData(partial, averageOptionFiled);
            long tp = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(tpObj);
            long totalActual = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(totalActualObj);
            long totalPredicted = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(totalPredictedObj);
            double beta = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(betaObj);
            String average = PrimitiveObjectInspectorFactory.writableStringObjectInspector.getPrimitiveJavaObject(averageObj);

            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            myAggr.merge(tp, totalActual, totalPredicted, beta, average);
        }

        @Override
        public DoubleWritable terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            FMeasureAggregationBuffer myAggr = (FMeasureAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }
    }

    public static class FMeasureAggregationBuffer extends
            GenericUDAFEvaluator.AbstractAggregationBuffer {
        long tp;
        /** tp + fn */
        long totalActual;
        /** tp + fp */
        long totalPredicted;
        double beta;
        String average;

        public FMeasureAggregationBuffer() {
            super();
        }

        void setOptions(double beta, String average) {
            this.beta = beta;
            this.average = average;
        }

        void reset() {
            this.tp = 0L;
            this.totalActual = 0L;
            this.totalPredicted = 0L;
        }

        void merge(long o_tp, long o_actual, long o_predicted, double beta, String average) {
            tp += o_tp;
            totalActual += o_actual;
            totalPredicted += o_predicted;
            this.beta = beta;
            this.average = average;
        }

        double get() {
            double squareBeta = beta * beta;
            double divisor;
            double numerator;

            if (average.equals("micro")) {
                divisor = denom(tp, totalActual, totalPredicted, squareBeta);
                numerator = (1.d + squareBeta) * tp;
            } else { // binary
                double precision = precision(tp, totalPredicted);
                double recall = recall(tp, totalActual);
                divisor = squareBeta * precision + recall;
                numerator = (1.d + squareBeta) * precision * recall;
            }

            if (divisor > 0) {
                return (numerator / divisor);
            } else {
                return -1.d;
            }
        }

        private static double denom(long tp, long totalActual, long totalPredicted,
                double squareBeta) {
            long lp = totalPredicted - tp;

            if (lp < 0) {
                lp = 0L;
            }

            long pl = totalActual - tp;

            if (pl < 0) {
                pl = 0L;
            }

            return squareBeta * (tp + lp) + tp + pl;
        }

        private static double precision(long tp, long totalPredicted) {
            return (totalPredicted == 0L) ? 0d : tp / (double) totalPredicted;
        }

        private static double recall(long tp, long totalActual) {
            return (totalActual == 0L) ? 0d : tp / (double) totalActual;
        }

        void iterate(@Nonnull List<?> actual, @Nonnull List<?> predicted) {
            final int numActual = actual.size();
            final int numPredicted = predicted.size();
            int countTp = 0;

            for (Object p : predicted) {
                if (actual.contains(p)) {
                    countTp++;
                }
            }
            this.tp += countTp;
            this.totalActual += numActual;
            this.totalPredicted += numPredicted;
        }
    }
}
