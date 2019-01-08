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
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

// @formatter:off
@Description(name = "array_slice",
        value = "_FUNC_(array<ANY> values, int offset [, int length]) - Slices the given array by the given offset and length parameters.",
        extended = "SELECT \n" + 
                "  array_slice(array(1,2,3,4,5,6),2,4),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   0, -- offset\n" + 
                "   2 -- length\n" + 
                "  ),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   6, -- offset\n" + 
                "   3 -- length\n" + 
                "  ),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   6, -- offset\n" + 
                "   10 -- length\n" + 
                "  ),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   6 -- offset\n" + 
                "  ),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   -3 -- offset\n" + 
                "  ),\n" + 
                "  array_slice(\n" + 
                "   array(\"zero\", \"one\", \"two\", \"three\", \"four\", \"five\", \"six\", \"seven\", \"eight\", \"nine\", \"ten\"),\n" + 
                "   -3, -- offset\n" + 
                "   2 -- length\n" + 
                "  );\n" + 
                "\n" + 
                " [3,4]\n" + 
                " [\"zero\",\"one\"] \n" + 
                " [\"six\",\"seven\",\"eight\"]\n" + 
                " [\"six\",\"seven\",\"eight\",\"nine\",\"ten\"]\n" + 
                " [\"six\",\"seven\",\"eight\",\"nine\",\"ten\"]\n" + 
                " [\"eight\",\"nine\",\"ten\"]\n" + 
                " [\"eight\",\"nine\"]")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ArraySliceUDF extends GenericUDF {

    private ListObjectInspector valuesOI;
    private PrimitiveObjectInspector offsetOI;
    @Nullable
    private PrimitiveObjectInspector lengthOI;

    private final List<Object> result = new ArrayList<>();

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentLengthException(
                "Expected 2 or 3 arguments, but got " + argOIs.length);
        }

        this.valuesOI = HiveUtils.asListOI(argOIs[0]);
        this.offsetOI = HiveUtils.asIntegerOI(argOIs[1]);
        if (argOIs.length == 3) {
            this.lengthOI = HiveUtils.asIntegerOI(argOIs[2]);
        }

        ObjectInspector elemOI = valuesOI.getListElementObjectInspector();
        return ObjectInspectorFactory.getStandardListObjectInspector(elemOI);
    }

    @Nullable
    @Override
    public List<Object> evaluate(@Nonnull DeferredObject[] args) throws HiveException {
        Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }
        final int size = valuesOI.getListLength(arg0);

        result.clear();

        Object arg1 = args[1].get();
        if (arg1 == null) {
            throw new UDFArgumentException("Offset argument MUST NOT be null");
        }

        final int offset = PrimitiveObjectInspectorUtils.getInt(arg1, offsetOI);
        final int fromIndex = (offset < 0) ? size + offset : offset;

        final int toIndex;
        if (args.length == 3) {
            Object arg2 = args[2].get();
            if (arg2 == null) {
                toIndex = size;
            } else {
                final int length = PrimitiveObjectInspectorUtils.getInt(arg2, lengthOI);
                if (length < 0) {
                    toIndex = size + length;
                } else {
                    toIndex = Math.min(size, fromIndex + length);
                }
            }
        } else {
            toIndex = size;
        }

        if (!validRange(fromIndex, toIndex, size)) {
            return null;
        }

        for (int i = fromIndex; i < toIndex; i++) {
            Object e = valuesOI.getListElement(arg0, i);
            result.add(e);
        }

        return result;
    }

    private static boolean validRange(final int fromIndex, final int toIndex, final int size)
            throws HiveException {
        if (fromIndex < 0) {
            return false;
        }
        if (toIndex < 0) {
            return false;
        }
        if (toIndex > size) {
            return false;
        }
        if (fromIndex > toIndex) {
            return false;
        }
        return true;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "array_slice(" + StringUtils.join(args, ',') + ")";
    }

}
