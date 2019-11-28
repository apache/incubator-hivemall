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

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;

@Description(name = "argmax",
        value = "_FUNC_(array<T> a) - Returns the first index of the maximum value",
        extended = "SELECT argmax(array(5,2,0,1));\n" + "0")
@UDFType(deterministic = true, stateful = false)
public final class ArgmaxUDF extends GenericUDF {

    private ListObjectInspector listOI;
    private PrimitiveObjectInspector elemOI;

    private IntWritable result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException("argmax takes exactly one argument: " + argOIs.length);
        }
        this.listOI = HiveUtils.asListOI(argOIs[0]);
        this.elemOI = HiveUtils.asPrimitiveObjectInspector(listOI.getListElementObjectInspector());

        this.result = new IntWritable();

        return PrimitiveObjectInspectorFactory.writableIntObjectInspector;
    }

    @Override
    public IntWritable evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        int index = -1;
        Object maxObject = null;
        final int size = listOI.getListLength(arg0);
        for (int i = 0; i < size; i++) {
            Object ai = listOI.getListElement(arg0, i);
            if (ai == null) {
                continue;
            }

            if (maxObject == null) {
                maxObject = ai;
                index = i;
            } else {
                final int cmp = ObjectInspectorUtils.compare(ai, elemOI, maxObject, elemOI);
                if (cmp > 0) {
                    maxObject = ai;
                    index = i;
                }
            }
        }

        result.set(index);
        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "argmax(" + StringUtils.join(children, ',') + ')';
    }

}
