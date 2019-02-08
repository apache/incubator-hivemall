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
import hivemall.utils.lang.Preconditions;

import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

//@formatter:off
@Description(name = "merge_maps",
        value = "_FUNC_(Map x) - Returns a map which contains the union of an aggregation of maps."
                + " Note that an existing value of a key can be replaced with the other duplicate key entry.",
        extended = "SELECT \n" + 
                "  merge_maps(m) \n" + 
                "FROM (\n" + 
                "  SELECT map('A',10,'B',20,'C',30) \n" + 
                "  UNION ALL \n" + 
                "  SELECT map('A',10,'B',20,'C',30)\n" + 
                ") t")
//@formatter:on
public final class MergeMapsUDAF extends AbstractGenericUDAFResolver {

    @Override
    public MergeMapsEvaluator getEvaluator(TypeInfo[] types) throws SemanticException {
        if (types.length != 1) {
            throw new UDFArgumentTypeException(types.length - 1,
                "One argument is expected but got " + types.length);
        }
        TypeInfo paramType = types[0];
        if (paramType.getCategory() != Category.MAP) {
            throw new UDFArgumentTypeException(0, "Only maps supported for now ");
        }
        return new MergeMapsEvaluator();
    }

    public static final class MergeMapsEvaluator extends GenericUDAFEvaluator {

        private transient MapObjectInspector inputMapOI, mergeMapOI;
        private transient ObjectInspector inputKeyOI, inputValOI;

        @AggregationType(estimable = false)
        static final class MapAggBuffer extends AbstractAggregationBuffer {
            @Nonnull
            final Map<Object, Object> collectMap = new HashMap<Object, Object>();
        }

        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            Preconditions.checkArgument(parameters.length == 1);
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.inputMapOI = HiveUtils.asMapOI(parameters[0]);
                this.inputKeyOI = inputMapOI.getMapKeyObjectInspector();
                this.inputValOI = inputMapOI.getMapValueObjectInspector();
            } else {// from partial aggregation
                this.mergeMapOI = HiveUtils.asMapOI(parameters[0]);
                this.inputKeyOI = mergeMapOI.getMapKeyObjectInspector();
                this.inputValOI = mergeMapOI.getMapValueObjectInspector();
            }

            return ObjectInspectorFactory.getStandardMapObjectInspector(
                ObjectInspectorUtils.getStandardObjectInspector(inputKeyOI),
                ObjectInspectorUtils.getStandardObjectInspector(inputValOI));
        }

        @Override
        public MapAggBuffer getNewAggregationBuffer() throws HiveException {
            MapAggBuffer buff = new MapAggBuffer();
            reset(buff);
            return buff;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer buff)
                throws HiveException {
            MapAggBuffer aggrBuf = (MapAggBuffer) buff;
            aggrBuf.collectMap.clear();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            Preconditions.checkArgument(parameters.length == 1);

            Object param0 = parameters[0];
            if (param0 == null) {
                return;
            }

            Map<?, ?> m = inputMapOI.getMap(param0);
            MapAggBuffer myagg = (MapAggBuffer) agg;
            putIntoSet(m, myagg.collectMap, inputMapOI);
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            MapAggBuffer myagg = (MapAggBuffer) agg;
            Map<?, ?> m = mergeMapOI.getMap(partial);
            putIntoSet(m, myagg.collectMap, mergeMapOI);
        }

        private static void putIntoSet(@Nonnull final Map<?, ?> m,
                @Nonnull final Map<Object, Object> dst, @Nonnull final MapObjectInspector mapOI) {
            final ObjectInspector keyOI = mapOI.getMapKeyObjectInspector();
            final ObjectInspector valueOI = mapOI.getMapValueObjectInspector();

            for (Map.Entry<?, ?> e : m.entrySet()) {
                Object k = e.getKey();
                Object v = e.getValue();
                Object keyCopy = ObjectInspectorUtils.copyToStandardObject(k, keyOI);
                Object valCopy = ObjectInspectorUtils.copyToStandardObject(v, valueOI);
                dst.put(keyCopy, valCopy);
            }
        }

        @Override
        @Nonnull
        public Map<Object, Object> terminatePartial(
                @SuppressWarnings("deprecation") AggregationBuffer agg) throws HiveException {
            MapAggBuffer myagg = (MapAggBuffer) agg;
            return myagg.collectMap;
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MapAggBuffer myagg = (MapAggBuffer) agg;
            return myagg.collectMap;
        }

    }

}
