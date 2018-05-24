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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;

@Description(name = "array_append",
        value = "_FUNC_(array<T> arr, T elem) - Append an element to the end of an array",
        extended = "SELECT array_append(array(1,2),3);\n 1,2,3\n\n"
                + "SELECT array_append(array('a','b'),'c');\n \"a\",\"b\",\"c\"")
@UDFType(deterministic = true, stateful = false)
public final class ArrayAppendUDF extends GenericUDF {

    private ListObjectInspector listInspector;
    private PrimitiveObjectInspector listElemInspector;
    private PrimitiveObjectInspector primInspector;
    private boolean returnWritables;

    private final List<Object> ret = new ArrayList<Object>();

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        this.listInspector = HiveUtils.asListOI(argOIs[0]);
        this.listElemInspector =
                HiveUtils.asPrimitiveObjectInspector(listInspector.getListElementObjectInspector());
        this.primInspector = HiveUtils.asPrimitiveObjectInspector(argOIs[1]);
        if (listElemInspector.getPrimitiveCategory() != primInspector.getPrimitiveCategory()) {
            throw new UDFArgumentException(
                "array_append expects the list type to match the type of the value being appended");
        }
        this.returnWritables = listElemInspector.preferWritable();
        return ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorUtils.getStandardObjectInspector(listElemInspector));
    }

    @Nullable
    @Override
    public List<Object> evaluate(@Nonnull DeferredObject[] args) throws HiveException {
        ret.clear();

        Object arg0 = args[0].get();
        if (arg0 == null) {
            Object arg1 = args[1].get();
            if (arg1 != null) {
                Object toAppend = returnWritables ? primInspector.getPrimitiveWritableObject(arg1)
                        : primInspector.getPrimitiveJavaObject(arg1);
                return Arrays.asList(toAppend);
            }
            return null;
        }

        final int size = listInspector.getListLength(arg0);
        for (int i = 0; i < size; i++) {
            Object rawElem = listInspector.getListElement(arg0, i);
            if (rawElem == null) {
                continue;
            }
            Object obj = returnWritables ? listElemInspector.getPrimitiveWritableObject(rawElem)
                    : listElemInspector.getPrimitiveJavaObject(rawElem);
            ret.add(obj);
        }

        Object arg1 = args[1].get();
        if (arg1 != null) {
            Object toAppend = returnWritables ? primInspector.getPrimitiveWritableObject(arg1)
                    : primInspector.getPrimitiveJavaObject(arg1);
            ret.add(toAppend);
        }

        return ret;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "array_append(" + args[0] + ", " + args[1] + ")";
    }

}
