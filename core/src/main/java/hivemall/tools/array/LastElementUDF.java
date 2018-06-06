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

/**
 * Return the last element in an array.
 */
@Description(name = "last_element", value = "_FUNC_(x) - Return the last element in an array",
        extended = "SELECT last_element(array('a','b','c'));\n c")
@UDFType(deterministic = true, stateful = false)
public class LastElementUDF extends GenericUDF {

    private ListObjectInspector listInspector;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException("last_element takes an array as an argument");
        }
        listInspector = HiveUtils.asListOI(argOIs[0]);

        return listInspector.getListElementObjectInspector();
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        Object list = args[0].get();
        if (list == null) {
            return null;
        }

        final int lastIdx = listInspector.getListLength(list) - 1;
        if (lastIdx >= 0) {
            return listInspector.getListElement(list, lastIdx);
        } else {
            return null;
        }
    }

    @Override
    public String getDisplayString(String[] args) {
        return "last_element( " + args[0] + " )";
    }

}
