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

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;

@Description(name = "element_at",
        value = "_FUNC_(array<T> list, int pos) - Returns an element at the given position",
        extended = "SELECT element_at(array(1,2,3,4),0);\n 1\n\n"
                + "SELECT element_at(array(1,2,3,4),-2);\n 3")
@UDFType(deterministic = true, stateful = false)
public final class ArrayElementAtUDF extends GenericUDF {
    private ListObjectInspector listInspector;
    private IntObjectInspector intInspector;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException("element_at takes an array and an int as arguments");
        }
        this.listInspector = HiveUtils.asListOI(argOIs[0]);
        this.intInspector = HiveUtils.asIntOI(argOIs[1]);

        return listInspector.getListElementObjectInspector();
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        Object list = args[0].get();
        if (list == null) {
            return null;
        }
        Object arg1 = args[1].get();
        if (arg1 == null) {
            throw new HiveException("Index MUST not be null");
        }
        final int arrayLength = listInspector.getListLength(list);

        int idx = intInspector.get(arg1);
        if (idx < 0) {
            idx = arrayLength + idx;
            if (idx < 0) {
                return null;
            }
        } else if (idx >= arrayLength) {
            return null; // IndexOutOfBound
        }

        return listInspector.getListElement(list, idx);
    }

    @Override
    public String getDisplayString(String[] args) {
        return "element_at( " + args[0] + " , " + args[1] + " )";
    }

}
