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
package hivemall.tools.map;

import hivemall.utils.collections.maps.BoundedSortedMap;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.IntWritable;

/**
 * Convert two aggregated columns into a sorted key-value map.
 */
@Description(name = "to_ordered_map",
        value = "_FUNC_(key, value [, const int k|const boolean reverseOrder=false]) "
                + "- Convert two aggregated columns into an ordered key-value map")
public final class UDAFToOrderedMap extends UDAFToMap {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        @SuppressWarnings("deprecation")
        final TypeInfo[] typeInfo = info.getParameters();
        if (typeInfo.length != 2 && typeInfo.length != 3) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "Expecting two or three arguments: " + typeInfo.length);
        }
        if (typeInfo[0].getCategory() != ObjectInspector.Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0,
                "Only primitive type arguments are accepted for the key but "
                        + typeInfo[0].getTypeName() + " was passed as parameter 1.");
        }

        boolean reverseOrder = false;
        int size = 0;
        if (typeInfo.length == 3) {
            ObjectInspector[] argOIs = info.getParameterObjectInspectors();
            ObjectInspector argOI2 = argOIs[2];
            if (HiveUtils.isConstBoolean(argOI2)) {
                reverseOrder = HiveUtils.getConstBoolean(argOI2);
            } else if (HiveUtils.isConstInteger(argOI2)) {
                size = HiveUtils.getConstInt(argOI2);
                if (size == 0) {
                    throw new UDFArgumentException("Map size must be non-zero value: " + size);
                }
                reverseOrder = (size > 0); // positive size => top-k
            } else {
                throw new UDFArgumentTypeException(2,
                    "The third argument must be boolean or int type: " + typeInfo[2].getTypeName());
            }
        }

        if (reverseOrder) { // descending
            if (size == 0) {
                return new ReverseOrderedMapEvaluator();
            } else {
                return new TopKOrderedMapEvaluator();
            }
        } else { // ascending
            if (size == 0) {
                return new NaturalOrderedMapEvaluator();
            } else {
                return new TailKOrderedMapEvaluator();
            }
        }
    }

    public static class NaturalOrderedMapEvaluator extends UDAFToMapEvaluator {

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            ((MapAggregationBuffer) agg).container = new TreeMap<Object, Object>();
        }

    }

    public static class ReverseOrderedMapEvaluator extends UDAFToMapEvaluator {

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            ((MapAggregationBuffer) agg).container = new TreeMap<Object, Object>(
                Collections.reverseOrder());
        }

    }

    public static class TopKOrderedMapEvaluator extends GenericUDAFEvaluator {

        protected PrimitiveObjectInspector inputKeyOI;
        protected ObjectInspector inputValueOI;
        protected MapObjectInspector partialMapOI;
        protected PrimitiveObjectInspector sizeOI;

        protected StructObjectInspector internalMergeOI;

        protected StructField partialMapField;
        protected StructField sizeField;

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] argOIs) throws HiveException {
            super.init(mode, argOIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.inputKeyOI = HiveUtils.asPrimitiveObjectInspector(argOIs[0]);
                this.inputValueOI = argOIs[1];
                this.sizeOI = HiveUtils.asIntegerOI(argOIs[2]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) argOIs[0];
                this.internalMergeOI = soi;

                this.partialMapField = soi.getStructFieldRef("partialMap");
                // re-extract input key/value OIs
                MapObjectInspector partialMapOI = (MapObjectInspector) partialMapField.getFieldObjectInspector();
                this.inputKeyOI = HiveUtils.asPrimitiveObjectInspector(partialMapOI.getMapKeyObjectInspector());
                this.inputValueOI = partialMapOI.getMapValueObjectInspector();

                this.partialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                    ObjectInspectorUtils.getStandardObjectInspector(inputKeyOI),
                    ObjectInspectorUtils.getStandardObjectInspector(inputValueOI));

                this.sizeField = soi.getStructFieldRef("size");
                this.sizeOI = (PrimitiveObjectInspector) sizeField.getFieldObjectInspector();
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI(inputKeyOI, inputValueOI);
            } else {// terminate
                outputOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                    ObjectInspectorUtils.getStandardObjectInspector(inputKeyOI),
                    ObjectInspectorUtils.getStandardObjectInspector(inputValueOI));
            }
            return outputOI;
        }

        @Nonnull
        private static StructObjectInspector internalMergeOI(
                @Nonnull PrimitiveObjectInspector keyOI, @Nonnull ObjectInspector valueOI) {
            List<String> fieldNames = new ArrayList<String>();
            List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("partialMap");
            fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                ObjectInspectorUtils.getStandardObjectInspector(keyOI),
                ObjectInspectorUtils.getStandardObjectInspector(valueOI)));

            fieldNames.add("size");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        static class MapAggregationBuffer extends AbstractAggregationBuffer {
            @Nullable
            Map<Object, Object> container;
            int size;

            MapAggregationBuffer() {
                super();
            }
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;
            myagg.container = null;
            myagg.size = 0;
        }

        @Override
        public MapAggregationBuffer getNewAggregationBuffer() throws HiveException {
            MapAggregationBuffer myagg = new MapAggregationBuffer();
            reset(myagg);
            return myagg;
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            assert (parameters.length == 3);
            if (parameters[0] == null) {
                return;
            }

            Object key = ObjectInspectorUtils.copyToStandardObject(parameters[0], inputKeyOI);
            Object value = ObjectInspectorUtils.copyToStandardObject(parameters[1], inputValueOI);
            int size = Math.abs(HiveUtils.getInt(parameters[2], sizeOI)); // size could be negative for tail-k

            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;
            if (myagg.container == null) {
                initBuffer(myagg, size);
            }
            myagg.container.put(key, value);
        }

        void initBuffer(@Nonnull MapAggregationBuffer agg, @Nonnegative int size) {
            Preconditions.checkArgument(size > 0, "size MUST be greather than zero: " + size);

            agg.container = new BoundedSortedMap<Object, Object>(size, true);
            agg.size = size;
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;

            Object[] partialResult = new Object[2];
            partialResult[0] = myagg.container;
            partialResult[1] = new IntWritable(myagg.size);

            return partialResult;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;

            Object partialMapObj = internalMergeOI.getStructFieldData(partial, partialMapField);
            Map<?, ?> partialMap = partialMapOI.getMap(HiveUtils.castLazyBinaryObject(partialMapObj));
            if (partialMap == null) {
                return;
            }

            if (myagg.container == null) {
                Object sizeObj = internalMergeOI.getStructFieldData(partial, sizeField);
                int size = HiveUtils.getInt(sizeObj, sizeOI);
                initBuffer(myagg, size);
            }
            for (Map.Entry<?, ?> e : partialMap.entrySet()) {
                Object key = ObjectInspectorUtils.copyToStandardObject(e.getKey(), inputKeyOI);
                Object value = ObjectInspectorUtils.copyToStandardObject(e.getValue(), inputValueOI);
                myagg.container.put(key, value);
            }
        }

        @Override
        @Nullable
        public Map<Object, Object> terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;
            return myagg.container;
        }

    }

    public static class TailKOrderedMapEvaluator extends TopKOrderedMapEvaluator {

        @Override
        void initBuffer(MapAggregationBuffer agg, int size) {
            agg.container = new BoundedSortedMap<Object, Object>(size);
            agg.size = size;
        }
    }

}
