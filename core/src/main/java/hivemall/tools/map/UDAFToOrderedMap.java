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

import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardMapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.IntWritable;

/**
 * Convert two aggregated columns into a sorted key-value map.
 */
@Description(name = "to_ordered_map",
        value = "_FUNC_(key, value [, const boolean|int reverseOrder=false|size]) "
                + "- Convert two aggregated columns into an ordered key-value map")
public class UDAFToOrderedMap extends UDAFToMap {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        @SuppressWarnings("deprecation")
        TypeInfo[] typeInfo = info.getParameters();
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
            if (HiveUtils.isBooleanTypeInfo(typeInfo[2])) {
                reverseOrder = HiveUtils.getConstBoolean(argOIs[2]);
            } else if (HiveUtils.isIntegerTypeInfo(typeInfo[2])) {
                size = HiveUtils.getConstInt(argOIs[2]);
                if (size == 0) {
                    throw new UDFArgumentException("Map size must be nonzero: " + size);
                }
                reverseOrder = (size > 0); // positive size => top-k
            } else {
                throw new UDFArgumentTypeException(2, "The third argument must be boolean or integer type: "
                        + typeInfo[2].getTypeName());
            }
        }

        if (reverseOrder) { // descending
            if (size != 0) {
                return new TopKOrderedMapEvaluator();
            }
            return new ReverseOrderedMapEvaluator();
        } else { // ascending
            if (size != 0) {
                return new TailKOrderedMapEvaluator();
            }
            return new NaturalOrderedMapEvaluator();
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
        protected StandardMapObjectInspector partialMapOI;
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
                StandardMapObjectInspector partialMapOI = (StandardMapObjectInspector) partialMapField.getFieldObjectInspector();
                this.inputKeyOI = HiveUtils.asPrimitiveObjectInspector(partialMapOI.getMapKeyObjectInspector());
                this.inputValueOI = partialMapOI.getMapValueObjectInspector();

                this.sizeField = soi.getStructFieldRef("size");
                this.sizeOI = (PrimitiveObjectInspector) sizeField.getFieldObjectInspector();
            }

            this.partialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                    ObjectInspectorUtils.getStandardObjectInspector(inputKeyOI),
                    ObjectInspectorUtils.getStandardObjectInspector(inputValueOI));

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                ArrayList<String> fieldNames = new ArrayList<String>();
                ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

                fieldNames.add("partialMap");
                fieldOIs.add(partialMapOI);

                fieldNames.add("size");
                fieldOIs.add(sizeOI);

                outputOI = ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames,
                        fieldOIs);
            } else {// terminate
                outputOI = partialMapOI;
            }
            return outputOI;
        }

        static class MapAggregationBuffer extends AbstractAggregationBuffer {
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
            myagg.container = new TreeMap<Object, Object>(Collections.reverseOrder());
            myagg.size = Integer.MAX_VALUE;
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
            myagg.container.put(key, value);
            myagg.size = size;
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
            for (Map.Entry<?, ?> e : partialMap.entrySet()) {
                Object key = ObjectInspectorUtils.copyToStandardObject(e.getKey(), inputKeyOI);
                Object value = ObjectInspectorUtils.copyToStandardObject(e.getValue(), inputValueOI);
                myagg.container.put(key, value);
            }

            Object sizeObj = internalMergeOI.getStructFieldData(partial, sizeField);
            int size = HiveUtils.getInt(sizeObj, sizeOI);
            myagg.size = size;
        }

        @Override
        public Map<Object, Object> terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;
            if (myagg.size < myagg.container.size()) {
                Object toKey = myagg.container.keySet().toArray()[myagg.size];
                return ((SortedMap<Object, Object>) myagg.container).headMap(toKey);
            }
            return myagg.container;
        }

    }

    public static class TailKOrderedMapEvaluator extends TopKOrderedMapEvaluator {

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggregationBuffer myagg = (MapAggregationBuffer) agg;
            myagg.container = new TreeMap<Object, Object>();
            myagg.size = Integer.MAX_VALUE;
        }

    }

}
