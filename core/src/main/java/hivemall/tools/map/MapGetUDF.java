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

import hivemall.utils.lang.StringUtils;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters.Converter;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;

//@formatter:off
@Description(name = "map_get",
        value = "_FUNC_(MAP<K> a, K n) - Returns the value corresponding to the key in the map.",
        extended = "Note this is a workaround for a Hive issue that non-constant expression for map indexes not supported.\n" +
                "See https://issues.apache.org/jira/browse/HIVE-1955\n\n" +
                "WITH tmp as (\n" +
                "  SELECT \"one\" as key\n" +
                "  UNION ALL\n" +
                "  SELECT \"two\" as key\n" +
                ")\n" +
                "SELECT map_get(map(\"one\",1,\"two\",2),key)\n" +
                "FROM tmp;\n\n" +
                "> 1\n" +
                "> 2")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class MapGetUDF extends GenericUDF {

    private transient MapObjectInspector mapOI;
    private transient Converter converter;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        if (arguments.length != 2) {
            throw new UDFArgumentLengthException("map_get accepts exactly 2 arguments.");
        }

        if (arguments[0] instanceof MapObjectInspector) {
            this.mapOI = (MapObjectInspector) arguments[0];
        } else {
            throw new UDFArgumentTypeException(0,
                "\"map\" is expected for the first argument, but \"" + arguments[0].getTypeName()
                        + "\" is found");
        }

        // index has to be a primitive
        if (!(arguments[1] instanceof PrimitiveObjectInspector)) {
            throw new UDFArgumentTypeException(1,
                "Primitive Type is expected but " + arguments[1].getTypeName() + "\" is found");
        }

        PrimitiveObjectInspector inputOI = (PrimitiveObjectInspector) arguments[1];
        ObjectInspector indexOI =
                ObjectInspectorConverters.getConvertedOI(inputOI, mapOI.getMapKeyObjectInspector());
        this.converter = ObjectInspectorConverters.getConverter(inputOI, indexOI);

        return mapOI.getMapValueObjectInspector();
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        assert (arguments.length == 2);
        Object index = arguments[1].get();

        Object indexObject = converter.convert(index);
        if (indexObject == null) {
            return null;
        }

        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        return mapOI.getMapValueElement(arg0, indexObject);
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_get(" + StringUtils.join(children, ',') + ')';
    }

}
