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

import org.apache.hadoop.hive.ql.exec.Description;
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Convert two aggregated columns into a fixed-size sorted map.
 */
@Description(name = "to_bounded_ordered_map",
        value = "_FUNC_(key, value, size [, const boolean reverseOrder=false]) "
                + "- Convert two aggregated columns into a fixed-size sorted map")
public class UDAFToBoundedOrderedMap extends UDAFToMap {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        @SuppressWarnings("deprecation")
        TypeInfo[] typeInfo = info.getParameters();
        if (typeInfo.length != 3 && typeInfo.length != 4) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "Expecting three or four arguments: " + typeInfo.length);
        }

        if (typeInfo[0].getCategory() != ObjectInspector.Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0,
                "Only primitive type arguments are accepted for the key but "
                        + typeInfo[0].getTypeName() + " was passed as parameter 1.");
        }

        if (!HiveUtils.isIntegerTypeInfo(typeInfo[2])) {
            throw new UDFArgumentTypeException(2, "The third argument must be integer type: "
                    + typeInfo[2].getTypeName());
        }

        boolean reverseOrder = false;
        if (typeInfo.length == 4) {
            if (!HiveUtils.isBooleanTypeInfo(typeInfo[3])) {
                throw new UDFArgumentTypeException(3, "The fourth argument must be boolean type: "
                        + typeInfo[3].getTypeName());
            }
            ObjectInspector[] argOIs = info.getParameterObjectInspectors();
            reverseOrder = HiveUtils.getConstBoolean(argOIs[3]);
        }

        if (reverseOrder) {
            return new BoundedReverseOrderedMapEvaluator();
        } else {
            return new BoundedOrderedMapEvaluator();
        }
    }

    public static class BoundedOrderedMapEvaluator extends GenericUDAFEvaluator {

        protected PrimitiveObjectInspector inputKeyOI;
        protected ObjectInspector inputValueOI;
        protected StandardMapObjectInspector partialMapOI;
        protected PrimitiveObjectInspector sizeOI;

        protected StructObjectInspector internalMergeOI;

        protected StructField partialMapField;
        protected StructField sizeField;

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] argOIs) throws HiveException {
            assert (argOIs.length == 3) : argOIs.length;
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

        static class BoundedMapAggregationBuffer extends AbstractAggregationBuffer {
            Map<Object, Object> container;
            int size;

            BoundedMapAggregationBuffer() {
                super();
            }

            Map<Object, Object> get() {
                if (size < container.size()) {
                    Object toKey = container.keySet().toArray()[size];
                    return ((SortedMap<Object, Object>) container).headMap(toKey);
                }
                return container;
            }
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            BoundedMapAggregationBuffer myagg = (BoundedMapAggregationBuffer) agg;
            myagg.container = new TreeMap<Object, Object>();
            myagg.size = Integer.MAX_VALUE;
        }

        @Override
        public BoundedMapAggregationBuffer getNewAggregationBuffer() throws HiveException {
            BoundedMapAggregationBuffer myagg = new BoundedMapAggregationBuffer();
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
            int size = HiveUtils.getInt(parameters[2], sizeOI);

            BoundedMapAggregationBuffer myagg = (BoundedMapAggregationBuffer) agg;
            myagg.container.put(key, value);
            myagg.size = size;
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            BoundedMapAggregationBuffer myagg = (BoundedMapAggregationBuffer) agg;

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

            BoundedMapAggregationBuffer myagg = (BoundedMapAggregationBuffer) agg;

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
            return ((BoundedMapAggregationBuffer) agg).get();
        }

    }

    public static class BoundedReverseOrderedMapEvaluator extends BoundedOrderedMapEvaluator {

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            BoundedMapAggregationBuffer myagg = (BoundedMapAggregationBuffer) agg;
            myagg.container = new TreeMap<Object, Object>(Collections.reverseOrder());
            myagg.size = Integer.MAX_VALUE;
        }

    }
}
