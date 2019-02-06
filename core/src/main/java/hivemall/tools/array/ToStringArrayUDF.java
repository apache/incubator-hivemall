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
package hivemall.tools.array;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "to_string_array", value = "_FUNC_(array<ANY>) - Returns an array of strings",
        extended = "select to_string_array(array(1.0,2.0,3.0));\n\n" + "[\"1.0\",\"2.0\",\"3.0\"]")
@UDFType(deterministic = true, stateful = false)
public final class ToStringArrayUDF extends GenericUDF {

    private ListObjectInspector listOI;
    private List<String> result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException(
                "to_string_array expects exactly one argument: " + argOIs.length);
        }
        this.listOI = HiveUtils.asListOI(argOIs[0]);
        this.result = new ArrayList<>();

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector);
    }

    @Override
    public List<String> evaluate(DeferredObject[] arguments) throws HiveException {
        final Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        result.clear();

        final int len = listOI.getListLength(arg0);
        for (int i = 0; i < len; i++) {
            final Object e = listOI.getListElement(arg0, i);
            if (e == null) {
                result.add(null);
            } else {
                result.add(e.toString());
            }
        }

        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "to_string_array(" + StringUtils.join(children, ',') + ')';
    }

}
