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
import java.util.Set;
import java.util.TreeSet;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;

/**
 * Return a list of unique entries for a given set of lists.
 *
 * <pre>
 * {1, 2} ∪ {1, 2} = {1, 2},
 * {1, 2} ∪ {2, 3} = {1, 2, 3},
 * {1, 2, 3} ∪ {3, 4, 5} = {1, 2, 3, 4, 5}
 * </pre>
 */
@Description(name = "array_union",
        value = "_FUNC_(array1, array2, ...) - Returns the union of a set of arrays",
        extended = "SELECT array_union(array(1,2),array(1,2));\n" + "[1,2]\n\n"
                + "SELECT array_union(array(1,2),array(2,3),array(2,5));\n" + "[1,2,3,5]")
@UDFType(deterministic = true, stateful = false)
public final class ArrayUnionUDF extends GenericUDF {

    private ListObjectInspector[] _listOIs;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 2) {
            throw new UDFArgumentException("Expecting at least two arrays as arguments");
        }

        ListObjectInspector[] listOIs = new ListObjectInspector[argOIs.length];
        ListObjectInspector arg0OI = HiveUtils.asListOI(argOIs[0]);
        listOIs[0] = arg0OI;
        ObjectInspector arg0ElemOI = arg0OI.getListElementObjectInspector();

        for (int i = 1; i < argOIs.length; ++i) {
            ListObjectInspector checkOI = HiveUtils.asListOI(argOIs[i]);
            if (!ObjectInspectorUtils.compareTypes(arg0ElemOI,
                checkOI.getListElementObjectInspector())) {
                throw new UDFArgumentException("Array types does not match: " + arg0OI.getTypeName()
                        + " != " + checkOI.getTypeName());
            }
            listOIs[i] = checkOI;
        }

        this._listOIs = listOIs;

        return ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorUtils.getStandardObjectInspector(arg0ElemOI,
                ObjectInspectorCopyOption.WRITABLE));
    }

    @Override
    public List<Object> evaluate(DeferredObject[] args) throws HiveException {
        final Set<Object> objectSet = new TreeSet<Object>(); // new HashSet<Object>();

        for (int i = 0; i < args.length; ++i) {
            final Object undeferred = args[i].get();
            if (undeferred == null) {
                continue;
            }

            final ListObjectInspector oi = _listOIs[i];
            final ObjectInspector elemOI = oi.getListElementObjectInspector();

            for (int j = 0, len = oi.getListLength(undeferred); j < len; ++j) {
                Object nonStd = oi.getListElement(undeferred, j);
                Object copyed = ObjectInspectorUtils.copyToStandardObject(nonStd, elemOI,
                    ObjectInspectorCopyOption.WRITABLE);
                objectSet.add(copyed);
            }
        }

        return new ArrayList<>(objectSet);
    }

    @Override
    public String getDisplayString(String[] args) {
        return "array_union(" + args[0] + ", " + args[1] + " )";
    }

}
