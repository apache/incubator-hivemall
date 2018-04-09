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
import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;

@Description(name = "array_flatten",
        value = "_FUNC_(array<array<ANY>>) - Returns an array with the elements flattened.")
@UDFType(deterministic = true, stateful = false)
public final class ArrayFlattenUDF extends GenericUDF {

    private ListObjectInspector listOI;
    private ListObjectInspector nextedListOI;
    private ObjectInspector elemOI;

    private final List<Object> result = new ArrayList<>();

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException(
                "array_flatten expects exactly one argument: " + argOIs.length);
        }

        this.listOI = HiveUtils.asListOI(argOIs[0]);
        ObjectInspector listElemOI = listOI.getListElementObjectInspector();
        if (listElemOI.getCategory() != Category.LIST) {
            throw new UDFArgumentException(
                "array_flatten takes array of array for the argument: " + listOI.toString());
        }
        this.nextedListOI = HiveUtils.asListOI(listElemOI);
        this.elemOI = nextedListOI.getListElementObjectInspector();

        return ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorUtils.getStandardObjectInspector(elemOI));
    }

    @Override
    public List<Object> evaluate(DeferredObject[] args) throws HiveException {
        result.clear();

        Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }

        final int listLength = listOI.getListLength(arg0);
        for (int i = 0; i < listLength; i++) {
            final Object subarray = listOI.getListElement(arg0, i);
            if (subarray == null) {
                continue;
            }

            final int subarrayLength = nextedListOI.getListLength(subarray);
            for (int j = 0; j < subarrayLength; j++) {
                Object elem = nextedListOI.getListElement(subarray, j);
                if (elem == null) {
                    continue;
                }
                result.add(elem);
            }
        }

        return result;
    }

    @Override
    public String getDisplayString(String[] args) {
        final StringBuffer buf = new StringBuffer();
        buf.append("array_flatten(");
        for (int i = 0, len = args.length; i < len; i++) {
            if (i != 0) {
                buf.append(", ");
            }
            buf.append(args[i]);
        }
        buf.append(")");
        return buf.toString();
    }

}
