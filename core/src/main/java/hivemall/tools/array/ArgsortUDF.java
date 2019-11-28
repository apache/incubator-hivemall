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

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import javax.annotation.Nullable;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;

// @formatter:off
@Description(name = "argsort",
        value = "_FUNC_(array<ANY> a) - Returns the indices that would sort an array.",
        extended = "SELECT argsort(array(5,2,0,1));\n" + 
                "2, 3, 1, 0\n" + 
                "\n" + 
                "SELECT array_slice(array(5,2,0,1), argsort(array(5,2,0,1)));\n" + 
                "0, 1, 2, 5")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ArgsortUDF extends GenericUDF {

    private ListObjectInspector listOI;
    private ObjectInspector elemOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentLengthException(
                "argsort(array<ANY> a) takes exactly 1 argument: " + argOIs.length);
        }
        ObjectInspector argOI0 = argOIs[0];
        if (argOI0.getCategory() != Category.LIST) {
            throw new UDFArgumentException(
                "argsort(array<ANY> a) expects array<ANY> for the first argument: "
                        + argOI0.getTypeName());
        }

        this.listOI = HiveUtils.asListOI(argOI0);
        this.elemOI = listOI.getListElementObjectInspector();

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableIntObjectInspector);
    }

    @Nullable
    @Override
    public List<IntWritable> evaluate(DeferredObject[] arguments) throws HiveException {
        final Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        final int size = listOI.getListLength(arg0);

        final Integer[] indexes = new Integer[size];
        for (int i = 0; i < size; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                Object ei = listOI.getListElement(arg0, i.intValue());
                Object ej = listOI.getListElement(arg0, j.intValue());
                return ObjectInspectorUtils.compare(ei, elemOI, ej, elemOI);
            }
        });

        final IntWritable[] ret = new IntWritable[size];
        for (int i = 0; i < size; i++) {
            ret[i] = new IntWritable(indexes[i].intValue());
        }
        return Arrays.asList(ret);
    }

    @Override
    public String getDisplayString(String[] children) {
        return "argsort(" + StringUtils.join(children, ',') + ')';
    }

}
