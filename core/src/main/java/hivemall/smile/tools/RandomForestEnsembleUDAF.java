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
package hivemall.smile.tools;

import hivemall.tools.array.ArrayAvgGenericUDAF.Evaluator;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.SizeOf;

import java.util.ArrayList;
import java.util.List;

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
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.IntWritable;

@Description(
        name = "rf_ensemble",
        value = "_FUNC_(int yhat, double model_weight, array<double> posteriori)"
                + " - Returns emsebled prediction results in <int label, double probability, array<double> probabilities>")
public final class RandomForestEnsembleUDAF extends AbstractGenericUDAFResolver {

    private RandomForestEnsembleUDAF() {}// prevent instantiation

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 3) {
            throw new UDFArgumentLengthException("Expected 3 arguments but got " + typeInfo.length);
        }
        if (!HiveUtils.isIntegerTypeInfo(typeInfo[0])) {
            throw new UDFArgumentTypeException(0, "Expected INT for yhat: " + typeInfo[0]);
        }
        if (!HiveUtils.isFloatingPointTypeInfo(typeInfo[1])) {
            throw new UDFArgumentTypeException(1, "Expected DOUBLE or FLOAT for model_weight: "
                    + typeInfo[1]);
        }
        if (!HiveUtils.isFloatingPointListTypeInfo(typeInfo[2])) {
            throw new UDFArgumentTypeException(2, "ARRAY<double> is expected for posteriori: "
                    + typeInfo[2]);
        }
        return new Evaluator();
    }


    @SuppressWarnings("deprecation")
    public static final class RfEvaluator extends GenericUDAFEvaluator {

        private PrimitiveObjectInspector yhatOI;
        private PrimitiveObjectInspector weightOI;
        private ListObjectInspector posterioriOI;
        private PrimitiveObjectInspector posterioriElemOI;

        private StructObjectInspector internalMergeOI;
        private StructField sizeField, weightsField, posterioriField;
        private IntObjectInspector sizeFieldOI;
        private StandardListObjectInspector weightsFieldOI;
        private StandardListObjectInspector posterioriFieldOI;

        public RfEvaluator() {
            super();
        }

        @Override
        public ObjectInspector init(@Nonnull Mode mode, @Nonnull ObjectInspector[] parameters)
                throws HiveException {
            super.init(mode, parameters);
            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                Preconditions.checkArgument(parameters.length == 3, "Expected 3 arguments but got "
                        + parameters.length, HiveException.class);
                this.yhatOI = HiveUtils.asIntegerOI(parameters[0]);
                this.weightOI = HiveUtils.asDoubleCompatibleOI(parameters[1]);
                this.posterioriOI = HiveUtils.asListOI(parameters[2]);
                this.posterioriElemOI = HiveUtils.asDoubleCompatibleOI(posterioriOI.getListElementObjectInspector());
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.sizeField = soi.getStructFieldRef("size");
                this.weightsField = soi.getStructFieldRef("weights");
                this.posterioriField = soi.getStructFieldRef("posteriori");
                this.weightsFieldOI = ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
                this.posterioriFieldOI = ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                List<String> fieldNames = new ArrayList<>(3);
                List<ObjectInspector> fieldOIs = new ArrayList<>(3);
                fieldNames.add("size");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
                fieldNames.add("weights");
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
                fieldNames.add("posteriori");
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
                outputOI = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames,
                    fieldOIs);
            } else {// terminate
                List<String> fieldNames = new ArrayList<>(3);
                List<ObjectInspector> fieldOIs = new ArrayList<>(3);
                fieldNames.add("label");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
                fieldNames.add("probability");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
                fieldNames.add("probabilities");
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
                outputOI = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames,
                    fieldOIs);
            }
            return outputOI;
        }

        @Override
        public RfAggregationBuffer getNewAggregationBuffer() throws HiveException {
            RfAggregationBuffer buf = new RfAggregationBuffer();
            reset(buf);
            return buf;
        }

        @Override
        public void reset(AggregationBuffer agg) throws HiveException {
            RfAggregationBuffer buf = (RfAggregationBuffer) agg;
            buf.reset();
        }

        @Override
        public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
            RfAggregationBuffer buf = (RfAggregationBuffer) agg;

            Preconditions.checkNotNull(parameters[0]);
            int yhat = PrimitiveObjectInspectorUtils.getInt(parameters[0], yhatOI);
            Preconditions.checkNotNull(parameters[1]);
            double weight = PrimitiveObjectInspectorUtils.getDouble(parameters[1], weightOI);
            Preconditions.checkNotNull(parameters[2]);
            double[] posteriori = HiveUtils.asDoubleArray(parameters[2], posterioriOI,
                posterioriElemOI);

            buf.iterate(yhat, weight, posteriori);
        }

        @Override
        public Object terminatePartial(AggregationBuffer agg) throws HiveException {
            RfAggregationBuffer buf = (RfAggregationBuffer) agg;
            if (buf._weights == null) {
                return null;
            }

            Object[] partial = new Object[2];
            partial[0] = WritableUtils.toWritableList(buf._weights);
            partial[1] = WritableUtils.toWritableList(buf._posteriori);
            return partial;
        }

        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            if (partial == null) {
                return;
            }
            RfAggregationBuffer buf = (RfAggregationBuffer) agg;

            Object o1 = internalMergeOI.getStructFieldData(partial, sizeField);
            int size = sizeFieldOI.get(o1);
            Object weights = internalMergeOI.getStructFieldData(partial, weightsField);
            Object posteriori = internalMergeOI.getStructFieldData(partial, posterioriField);

            // --------------------------------------------------------------
            // [workaround]
            // java.lang.ClassCastException: org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray
            // cannot be cast to [Ljava.lang.Object;
            if (weights instanceof LazyBinaryArray) {
                weights = ((LazyBinaryArray) weights).getList();
            }
            if (posteriori instanceof LazyBinaryArray) {
                posteriori = ((LazyBinaryArray) posteriori).getList();
            }

            buf.merge(size, weights, posteriori, weightsFieldOI, posterioriFieldOI);
        }

        @Override
        public Object terminate(AggregationBuffer agg) throws HiveException {
            RfAggregationBuffer buf = (RfAggregationBuffer) agg;
            if (buf._k == -1) {
                return null;
            }

            int label = smile.math.Math.whichMax(buf._weights);
            double[] posteriori = buf._posteriori;
            smile.math.Math.unitize1(posteriori);
            double proba = posteriori[label];

            Object[] result = new Object[3];
            result[0] = new IntWritable(label);
            result[1] = new DoubleWritable(proba);
            result[2] = WritableUtils.toWritableList(posteriori);
            return result;
        }

    }

    public static final class RfAggregationBuffer extends AbstractAggregationBuffer {

        @Nullable
        private double[] _weights;
        @Nullable
        private double[] _posteriori;
        private int _k;

        public RfAggregationBuffer() {
            super();
        }

        void reset() {
            this._weights = null;
            this._posteriori = null;
            this._k = -1;
        }

        void iterate(final int yhat, final double weight, @Nonnull final double[] posteriori)
                throws HiveException {
            if (_weights == null) {
                this._k = posteriori.length;
                this._weights = new double[_k];
                this._posteriori = new double[_k];
            }
            if (yhat >= _k) {
                throw new HiveException("Predicted class " + yhat + " is out of bounds: " + _k);
            }
            if (posteriori.length != _k) {
                throw new HiveException("Given |posteriori| " + posteriori.length
                        + " is differs from expected one: " + _k);
            }

            _weights[yhat] += weight;
            _posteriori[yhat] += (posteriori[yhat] * weight);
        }

        void merge(int size, @Nonnull Object weightsObj, @Nonnull Object posterioriObj,
                @Nonnull StandardListObjectInspector weightsOI,
                @Nonnull StandardListObjectInspector posterioriOI) throws HiveException {

            if (size != _k) {
                if (size == -1) {
                    this._k = size;
                    this._weights = new double[size];
                    this._posteriori = new double[size];
                } else {
                    throw new HiveException("Mismatch in the number of elements");
                }
            }

            final double[] weights = _weights;
            final double[] posteriori = _posteriori;
            final DoubleObjectInspector doubleOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            for (int i = 0, len = _k; i < len; i++) {
                Object o1 = weightsOI.getListElement(weightsObj, i);
                weights[i] += doubleOI.get(o1);
                Object o2 = posterioriOI.getListElement(posterioriObj, i);
                posteriori[i] += doubleOI.get(o2);
            }
        }

        @Override
        public int estimate() {
            if (_k == -1) {
                return 0;
            }
            return _k * SizeOf.DOUBLE * 2;
        }

    }

}
