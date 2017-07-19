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

import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Counter;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.SizeOf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import javax.annotation.CheckForNull;
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
import org.apache.hadoop.hive.serde2.objectinspector.StandardMapObjectInspector;
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
        value = "_FUNC_(int yhat [, array<double> proba [, double model_weight=1.0]])"
                + " - Returns emsebled prediction results in <int label, double probability, array<double> probabilities>")
public final class RandomForestEnsembleUDAF extends AbstractGenericUDAFResolver {

    public RandomForestEnsembleUDAF() {
        super();
    }

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull final TypeInfo[] typeInfo)
            throws SemanticException {
        switch (typeInfo.length) {
            case 1: {
                if (!HiveUtils.isIntegerTypeInfo(typeInfo[0])) {
                    throw new UDFArgumentTypeException(0, "Expected INT for yhat: " + typeInfo[0]);
                }
                return new RfEvaluatorV1();
            }
            case 3:
                if (!HiveUtils.isFloatingPointTypeInfo(typeInfo[2])) {
                    throw new UDFArgumentTypeException(2,
                        "Expected DOUBLE or FLOAT for model_weight: " + typeInfo[2]);
                }
                /* fall through */
            case 2: {// typeInfo.length == 2 || typeInfo.length == 3
                if (!HiveUtils.isIntegerTypeInfo(typeInfo[0])) {
                    throw new UDFArgumentTypeException(0, "Expected INT for yhat: " + typeInfo[0]);
                }
                if (!HiveUtils.isFloatingPointListTypeInfo(typeInfo[1])) {
                    throw new UDFArgumentTypeException(1,
                        "ARRAY<double> is expected for posteriori: " + typeInfo[1]);
                }
                return new RfEvaluatorV2();
            }
            default:
                throw new UDFArgumentLengthException("Expected 1~3 arguments but got "
                        + typeInfo.length);
        }
    }

    @Deprecated
    public static final class RfEvaluatorV1 extends GenericUDAFEvaluator {

        // original input
        private PrimitiveObjectInspector yhatOI;

        // partial aggregation
        private StandardMapObjectInspector internalMergeOI;
        private IntObjectInspector keyOI;
        private IntObjectInspector valueOI;

        public RfEvaluatorV1() {
            super();
        }

        @Override
        public ObjectInspector init(@Nonnull Mode mode, @Nonnull ObjectInspector[] argOIs)
                throws HiveException {
            super.init(mode, argOIs);
            
            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.yhatOI = HiveUtils.asIntegerOI(argOIs[0]);
            } else {// from partial aggregation
                this.internalMergeOI = (StandardMapObjectInspector) argOIs[0];
                this.keyOI = HiveUtils.asIntOI(internalMergeOI.getMapKeyObjectInspector());
                this.valueOI = HiveUtils.asIntOI(internalMergeOI.getMapValueObjectInspector());
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial       
                outputOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector);
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
        public RfAggregationBufferV1 getNewAggregationBuffer() throws HiveException {
            RfAggregationBufferV1 buf = new RfAggregationBufferV1();
            buf.reset();
            return buf;
        }

        @Override
        public void reset(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV1 buf = (RfAggregationBufferV1) agg;
            buf.reset();
        }

        @Override
        public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
            RfAggregationBufferV1 buf = (RfAggregationBufferV1) agg;

            Preconditions.checkNotNull(parameters[0]);
            int yhat = PrimitiveObjectInspectorUtils.getInt(parameters[0], yhatOI);

            buf.iterate(yhat);
        }

        @Override
        public Object terminatePartial(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV1 buf = (RfAggregationBufferV1) agg;

            return buf.terminatePartial();
        }

        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            final RfAggregationBufferV1 buf = (RfAggregationBufferV1) agg;

            Map<?, ?> partialResult = internalMergeOI.getMap(partial);
            for (Map.Entry<?, ?> entry : partialResult.entrySet()) {
                putIntoMap(entry.getKey(), entry.getValue(), buf);
            }
        }

        private void putIntoMap(@CheckForNull Object key, @CheckForNull Object value,
                @Nonnull RfAggregationBufferV1 dst) {
            Preconditions.checkNotNull(key);
            Preconditions.checkNotNull(value);

            int k = keyOI.get(key);
            int v = valueOI.get(value);
            dst.merge(k, v);
        }

        @Override
        public Object terminate(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV1 buf = (RfAggregationBufferV1) agg;

            return buf.terminate();
        }

    }

    public static final class RfAggregationBufferV1 extends AbstractAggregationBuffer {

        @Nonnull
        private Counter<Integer> partial;

        public RfAggregationBufferV1() {
            super();
            reset();
        }

        void reset() {
            this.partial = new Counter<Integer>();
        }

        void iterate(final int k) {
            partial.increment(k);
        }

        @Nonnull
        Map<Integer, Integer> terminatePartial() {
            return partial.getMap();
        }

        void merge(final int k, final int v) {
            partial.increment(Integer.valueOf(k), v);
        }

        @Nullable
        Object[] terminate() {
            final Map<Integer, Integer> counts = partial.getMap();

            final int size = counts.size();
            if (size == 0) {
                return null;
            }

            final IntArrayList keyList = new IntArrayList(size);
            long totalCnt = 0L;
            Integer maxKey = null;
            int maxCnt = Integer.MIN_VALUE;
            for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
                Integer key = e.getKey();
                keyList.add(key);
                int cnt = e.getValue().intValue();
                totalCnt += cnt;
                if (cnt >= maxCnt) {
                    maxCnt = cnt;
                    maxKey = key;
                }
            }

            final int[] keyArray = keyList.toArray();
            Arrays.sort(keyArray);
            int last = keyArray[keyArray.length - 1];

            double totalCnt_d = (double) totalCnt;
            final double[] probabilities = new double[Math.max(2, last + 1)];
            for (int i = 0, len = probabilities.length; i < len; i++) {
                final Integer cnt = counts.get(Integer.valueOf(i));
                if (cnt == null) {
                    probabilities[i] = 0.d;
                } else {
                    probabilities[i] = cnt.intValue() / totalCnt_d;
                }
            }

            Object[] result = new Object[3];
            result[0] = new IntWritable(maxKey);
            double proba = maxCnt / totalCnt_d;
            result[1] = new DoubleWritable(proba);
            result[2] = WritableUtils.toWritableList(probabilities);
            return result;
        }

    }


    @SuppressWarnings("deprecation")
    public static final class RfEvaluatorV2 extends GenericUDAFEvaluator {

        private PrimitiveObjectInspector yhatOI;
        private ListObjectInspector posterioriOI;
        private PrimitiveObjectInspector posterioriElemOI;
        @Nullable
        private PrimitiveObjectInspector weightOI;

        private StructObjectInspector internalMergeOI;
        private StructField sizeField, posterioriField;
        private IntObjectInspector sizeFieldOI;
        private StandardListObjectInspector posterioriFieldOI;

        public RfEvaluatorV2() {
            super();
        }

        @Override
        public ObjectInspector init(@Nonnull Mode mode, @Nonnull ObjectInspector[] parameters)
                throws HiveException {
            super.init(mode, parameters);
            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.yhatOI = HiveUtils.asIntegerOI(parameters[0]);
                this.posterioriOI = HiveUtils.asListOI(parameters[1]);
                this.posterioriElemOI = HiveUtils.asDoubleCompatibleOI(posterioriOI.getListElementObjectInspector());
                if (parameters.length == 3) {
                    this.weightOI = HiveUtils.asDoubleCompatibleOI(parameters[2]);
                }
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.sizeField = soi.getStructFieldRef("size");
                this.posterioriField = soi.getStructFieldRef("posteriori");
                this.sizeFieldOI = PrimitiveObjectInspectorFactory.writableIntObjectInspector;
                this.posterioriFieldOI = ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                List<String> fieldNames = new ArrayList<>(3);
                List<ObjectInspector> fieldOIs = new ArrayList<>(3);
                fieldNames.add("size");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
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
        public RfAggregationBufferV2 getNewAggregationBuffer() throws HiveException {
            RfAggregationBufferV2 buf = new RfAggregationBufferV2();
            reset(buf);
            return buf;
        }

        @Override
        public void reset(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV2 buf = (RfAggregationBufferV2) agg;
            buf.reset();
        }

        @Override
        public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
            RfAggregationBufferV2 buf = (RfAggregationBufferV2) agg;

            Preconditions.checkNotNull(parameters[0]);
            int yhat = PrimitiveObjectInspectorUtils.getInt(parameters[0], yhatOI);
            Preconditions.checkNotNull(parameters[1]);
            double[] posteriori = HiveUtils.asDoubleArray(parameters[1], posterioriOI,
                posterioriElemOI);

            double weight = 1.0d;
            if (parameters.length == 3) {
                Preconditions.checkNotNull(parameters[2]);
                weight = PrimitiveObjectInspectorUtils.getDouble(parameters[2], weightOI);
            }

            buf.iterate(yhat, weight, posteriori);
        }

        @Override
        public Object terminatePartial(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV2 buf = (RfAggregationBufferV2) agg;
            if (buf._k == -1) {
                return null;
            }

            Object[] partial = new Object[2];
            partial[0] = new IntWritable(buf._k);
            partial[1] = WritableUtils.toWritableList(buf._posteriori);
            return partial;
        }

        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            if (partial == null) {
                return;
            }
            RfAggregationBufferV2 buf = (RfAggregationBufferV2) agg;

            Object o1 = internalMergeOI.getStructFieldData(partial, sizeField);
            int size = sizeFieldOI.get(o1);
            Object posteriori = internalMergeOI.getStructFieldData(partial, posterioriField);

            // --------------------------------------------------------------
            // [workaround]
            // java.lang.ClassCastException: org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray
            // cannot be cast to [Ljava.lang.Object;
            if (posteriori instanceof LazyBinaryArray) {
                posteriori = ((LazyBinaryArray) posteriori).getList();
            }

            buf.merge(size, posteriori, posterioriFieldOI);
        }

        @Override
        public Object terminate(AggregationBuffer agg) throws HiveException {
            RfAggregationBufferV2 buf = (RfAggregationBufferV2) agg;
            if (buf._k == -1) {
                return null;
            }

            double[] posteriori = buf._posteriori;
            int label = smile.math.Math.whichMax(posteriori);
            smile.math.Math.unitize1(posteriori);
            double proba = posteriori[label];

            Object[] result = new Object[3];
            result[0] = new IntWritable(label);
            result[1] = new DoubleWritable(proba);
            result[2] = WritableUtils.toWritableList(posteriori);
            return result;
        }

    }

    public static final class RfAggregationBufferV2 extends AbstractAggregationBuffer {

        @Nullable
        private double[] _posteriori;
        private int _k;

        public RfAggregationBufferV2() {
            super();
            reset();
        }

        void reset() {
            this._posteriori = null;
            this._k = -1;
        }

        void iterate(final int yhat, final double weight, @Nonnull final double[] posteriori)
                throws HiveException {
            if (_posteriori == null) {
                this._k = posteriori.length;
                this._posteriori = new double[_k];
            }
            if (yhat >= _k) {
                throw new HiveException("Predicted class " + yhat + " is out of bounds: " + _k);
            }
            if (posteriori.length != _k) {
                throw new HiveException("Given |posteriori| " + posteriori.length
                        + " is differs from expected one: " + _k);
            }

            _posteriori[yhat] += (posteriori[yhat] * weight);
        }

        void merge(int size, @Nonnull Object posterioriObj,
                @Nonnull StandardListObjectInspector posterioriOI) throws HiveException {

            if (size != _k) {
                if (_k == -1) {
                    this._k = size;
                    this._posteriori = new double[size];
                } else {
                    throw new HiveException("Mismatch in the number of elements: _k=" + _k
                            + ", size=" + size);
                }
            }

            final double[] posteriori = _posteriori;
            final DoubleObjectInspector doubleOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            for (int i = 0, len = _k; i < len; i++) {
                Object o2 = posterioriOI.getListElement(posterioriObj, i);
                posteriori[i] += doubleOI.get(o2);
            }
        }

        @Override
        public int estimate() {
            if (_k == -1) {
                return 0;
            }
            return SizeOf.INT + _k * SizeOf.DOUBLE;
        }

    }

}
