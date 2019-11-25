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

import java.util.Collections;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters.Converter;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;

//@formatter:off
@Description(name = "array_remove",
        value = "_FUNC_(array<PRIMITIVE> values, PRIMITIVE|array<PRIMITIVE> target)"
                + " - Returns an array that the target elements are removed from the original array",
        extended = "select array_remove(array(2.0,2.1,3.0,4.0,2.0),2), array_remove(array(2.0,3.0,4.0),array(3,2.0));\n" + 
                "[2.1,3,4]       [4]\n" + 
                "\n" + 
                "SELECT array_remove(array(1,null,3),null);\n" + 
                "[1,3]\n" + 
                "\n" + 
                "SELECT array_remove(array(1,null,3,null,5),null);\n" + 
                "[1,3,5]\n" + 
                "\n" + 
                "SELECT array_remove(array(1,null,3),array(null));\n" + 
                "[1,3]\n" + 
                "\n" + 
                "SELECT array_remove(array('aaa','bbb'),'bbb');\n" + 
                "[\"aaa\"]\n" + 
                "\n" + 
                "SELECT array_remove(array('aaa','bbb','ccc','bbb'), array('bbb','ccc'));\n" + 
                "[\"aaa\"]\n" + 
                "\n" + 
                "select array_remove(array(null),null);\n" + 
                "[]\n" + 
                "\n" + 
                "select array_remove(array(null,'bbb'),'aaa');\n" + 
                "[null,\"bbb\"]")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ArrayRemoveUDF extends GenericUDF {

    private ListObjectInspector valueListOI;
    private PrimitiveObjectInspector valueElemOI;
    private boolean isTargetList;
    @Nullable
    private ListObjectInspector targetListOI;
    private PrimitiveObjectInspector targetElemOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentLengthException("Expected 2 arguments, but got " + argOIs.length);
        }

        this.valueListOI = HiveUtils.asListOI(argOIs, 0);
        this.valueElemOI =
                HiveUtils.asPrimitiveObjectInspector(valueListOI.getListElementObjectInspector());

        if (HiveUtils.isListOI(argOIs[1])) {
            this.isTargetList = true;
            this.targetListOI = HiveUtils.asListOI(argOIs, 1);
            this.targetElemOI = HiveUtils.asPrimitiveObjectInspector(
                targetListOI.getListElementObjectInspector());
        } else {
            this.isTargetList = false;
            this.targetElemOI = HiveUtils.asPrimitiveObjectInspector(argOIs, 1);
        }

        return ObjectInspectorFactory.getStandardListObjectInspector(valueElemOI);
    }

    @Nullable
    @Override
    public Object evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        assert (arguments.length == 2);

        final List<?> values = HiveUtils.copyListObject(arguments[0], valueListOI);
        if (values == null) {
            return null;
        }

        final Object target = arguments[1].get();
        if (target == null) {
            values.removeAll(Collections.singletonList(null));
            return values;
        }

        if (isTargetList) {
            Converter converter = ObjectInspectorConverters.getConverter(targetListOI, valueListOI);
            removeAll(values, target, converter, valueListOI);
        } else {
            Converter converter = ObjectInspectorConverters.getConverter(targetElemOI, valueElemOI);
            removeAll(values, target, converter);
        }
        return values;
    }

    private static void removeAll(@Nonnull final List<?> values, @Nonnull final Object target,
            @Nonnull final Converter converter, @Nonnull final ListObjectInspector valueListOI) {
        Object converted = converter.convert(target);
        List<?> convertedList = valueListOI.getList(converted);
        values.removeAll(convertedList);
    }

    private static void removeAll(@Nonnull final List<?> values, @Nonnull final Object target,
            @Nonnull final Converter converter) {
        Object converted = converter.convert(target);
        values.removeAll(Collections.singleton(converted));
    }

    @Override
    public String getDisplayString(String[] children) {
        return "array_remove(" + StringUtils.join(children, ',') + ')';
    }

}
