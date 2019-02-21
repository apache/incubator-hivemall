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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;

@Description(name = "map_key_values",
        value = "_FUNC_(MAP<K, V> map) - "
                + "Returns a array of key-value pairs in array<named_struct<key,value>>",
        extended = "SELECT map_key_values(map(\"one\",1,\"two\",2));\n\n"
                + "> [{\"key\":\"one\",\"value\":1},{\"key\":\"two\",\"value\":2}]")
@UDFType(deterministic = true, stateful = false)
public final class MapKeyValuesUDF extends GenericUDF {

    private final ArrayList<Object[]> retArray = new ArrayList<Object[]>();

    private MapObjectInspector mapOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        if (arguments.length != 1) {
            throw new UDFArgumentLengthException(
                "The function MAP_KEYS only accepts one argument.");
        } else if (!(arguments[0] instanceof MapObjectInspector)) {
            throw new UDFArgumentTypeException(0,
                "\"" + Category.MAP.toString().toLowerCase()
                        + "\" is expected at function MAP_KEYS, " + "but \""
                        + arguments[0].getTypeName() + "\" is found");
        }

        this.mapOI = (MapObjectInspector) arguments[0];

        List<String> structFieldNames = new ArrayList<String>();
        List<ObjectInspector> structFieldObjectInspectors = new ArrayList<ObjectInspector>();
        structFieldNames.add("key");
        structFieldObjectInspectors.add(mapOI.getMapKeyObjectInspector());
        structFieldNames.add("value");
        structFieldObjectInspectors.add(mapOI.getMapValueObjectInspector());

        return ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorFactory.getStandardStructObjectInspector(structFieldNames,
                structFieldObjectInspectors));
    }

    @Override
    @Nullable
    public List<Object[]> evaluate(DeferredObject[] arguments) throws HiveException {
        Object mapObj = arguments[0].get();
        if (mapObj == null) {
            return null;
        }
        retArray.clear();
        final Map<?, ?> map = mapOI.getMap(mapObj);
        for (Map.Entry<?, ?> e : map.entrySet()) {
            retArray.add(new Object[] {e.getKey(), e.getValue()});
        }
        return retArray;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_key_values(" + StringUtils.join(children, ',') + ')';
    }

}
