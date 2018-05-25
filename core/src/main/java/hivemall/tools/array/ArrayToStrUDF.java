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

import javax.annotation.Nullable;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;

@Description(name = "array_to_str",
        value = "_FUNC_(array arr [, string sep=',']) - Convert array to string using a sperator",
        extended = "SELECT array_to_str(array(1,2,3),'-');\n" + "1-2-3")
@UDFType(deterministic = true, stateful = false)
public final class ArrayToStrUDF extends GenericUDF {

    private ListObjectInspector listOI;
    @Nullable
    private StringObjectInspector sepOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            throw new UDFArgumentLengthException(
                "array_to_str(array, string sep) expects one or two arguments: " + argOIs.length);
        }

        this.listOI = HiveUtils.asListOI(argOIs[0]);
        if (argOIs.length == 2) {
            this.sepOI = HiveUtils.asStringOI(argOIs[1]);
        }

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Override
    public String evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        String sep = ",";
        if (arguments.length == 2) {
            Object arg1 = arguments[1].get();
            if (arg1 != null) {
                sep = sepOI.getPrimitiveJavaObject(arg1);
            }
        }

        final StringBuilder buf = new StringBuilder();
        final int len = listOI.getListLength(arg0);
        for (int i = 0; i < len; i++) {
            Object e = listOI.getListElement(arg0, i);
            if (e != null) {
                if (i != 0 && buf.length() > 0) {
                    buf.append(sep);
                }
                buf.append(e.toString());
            }
        }
        return buf.toString();
    }

    @Override
    public String getDisplayString(String[] children) {
        return "array_to_str(" + StringUtils.join(children, ',') + ")";
    }

}
