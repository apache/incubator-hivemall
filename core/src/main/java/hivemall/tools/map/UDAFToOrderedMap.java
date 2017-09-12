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

import java.util.Collections;
import java.util.TreeMap;

import javax.annotation.Nonnegative;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;

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
            if (HiveUtils.isBooleanTypeInfo(typeInfo[2])) {
                reverseOrder = HiveUtils.getConstBoolean(argOIs[2]);
            } else if (HiveUtils.isIntegerTypeInfo(typeInfo[2])) {
                size = HiveUtils.getConstInt(argOIs[2]);
                if (size == 0) {
                    throw new UDFArgumentException("Map size must be nonzero: " + size);
                }
                reverseOrder = (size > 0); // positive size => top-k
                size = Math.abs(size);
            } else {
                throw new UDFArgumentTypeException(2,
                    "The third argument must be boolean or integer type: "
                            + typeInfo[2].getTypeName());
            }
        }

        if (reverseOrder) { // descending
            return new DescendingMapEvaluator(size);
        } else { // ascending
            return new AscendingMapEvaluator(size);
        }
    }

    public static final class AscendingMapEvaluator extends UDAFToMapEvaluator {

        private final int size;

        AscendingMapEvaluator(@Nonnegative int size) {
            super();
            this.size = size;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            if (size == 0) {
                ((MapAggregationBuffer) agg).container = new TreeMap<Object, Object>();
            } else {
                ((MapAggregationBuffer) agg).container = new BoundedSortedMap<Object, Object>(size);
            }
        }

    }

    public static final class DescendingMapEvaluator extends UDAFToMapEvaluator {

        private final int size;

        DescendingMapEvaluator(int size) {
            super();
            this.size = size;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            if (size == 0) {
                ((MapAggregationBuffer) agg).container = new TreeMap<Object, Object>(
                    Collections.reverseOrder());
            } else {
                ((MapAggregationBuffer) agg).container = new BoundedSortedMap<Object, Object>(size,
                    true);
            }
        }

    }
}
