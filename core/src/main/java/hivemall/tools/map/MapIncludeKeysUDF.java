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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;

@Description(name = "map_include_keys",
        value = "_FUNC_(Map<K,V> map, array<K> filteringKeys)"
                + " - Returns the filtered entries of a map having specified keys",
        extended = "SELECT map_include_keys(map(1,'one',2,'two',3,'three'),array(2,3));\n"
                + "{2:\"two\",3:\"three\"}")
@UDFType(deterministic = true, stateful = false)
public final class MapIncludeKeysUDF extends GenericUDF {

    private MapObjectInspector mapOI;
    private ListObjectInspector listOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentLengthException(
                "Expected two arguments for map_filter_keys: " + argOIs.length);
        }

        this.mapOI = HiveUtils.asMapOI(argOIs[0]);
        this.listOI = HiveUtils.asListOI(argOIs[1]);

        ObjectInspector mapKeyOI = mapOI.getMapKeyObjectInspector();
        ObjectInspector filterKeyOI = listOI.getListElementObjectInspector();
        if (!ObjectInspectorUtils.compareTypes(mapKeyOI, filterKeyOI)) {
            throw new UDFArgumentException("Element types does not match: mapKey "
                    + mapKeyOI.getTypeName() + ", filterKey" + filterKeyOI.getTypeName());
        }

        return ObjectInspectorUtils.getStandardObjectInspector(mapOI,
            ObjectInspectorCopyOption.WRITABLE);
    }

    @Override
    public Map<?, ?> evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }
        final Map<?, ?> map = (Map<?, ?>) ObjectInspectorUtils.copyToStandardObject(arg0, mapOI,
            ObjectInspectorCopyOption.WRITABLE);

        Object arg1 = arguments[1].get();
        if (arg1 == null) {
            return null;
        }
        final List<?> filterKeys = (List<?>) ObjectInspectorUtils.copyToStandardObject(arg1, listOI,
            ObjectInspectorCopyOption.WRITABLE);

        final Map<Object, Object> result = new HashMap<>();
        for (Object k : filterKeys) {
            Object v = map.get(k);
            if (v != null) {
                result.put(k, v);
            }
        }
        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_include_keys(" + StringUtils.join(children, ',') + ")";
    }

}
