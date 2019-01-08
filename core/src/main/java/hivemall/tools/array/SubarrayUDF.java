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

import static hivemall.utils.lang.StringUtils.join;

import hivemall.utils.hadoop.HiveUtils;

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
@Description(name = "subarray",
        value = "_FUNC_(array<ANY> values, int fromIndex [, int toIndex])" +
                "- Returns a slice of the original array between the inclusive fromIndex and the exclusive toIndex.",
        extended = "SELECT \n" + 
                "  subarray(array(0,1,2,3,4,5),4),\n" + 
                "  subarray(array(0,1,2,3,4,5),3,4),\n" + 
                "  subarray(array(0,1,2,3,4,5),3,3),\n" + 
                "  subarray(array(0,1,2,3,4,5),3,2),\n" + 
                "  subarray(array(0,1,2,3,4,5),0,2),\n" + 
                "  subarray(array(0,1,2,3,4,5),-1,2),\n" +  
                "  subarray(array(1,2,3,4,5,6),4),\n" + 
                "  subarray(array(1,2,3,4,5,6),4,6),\n" + 
                "  subarray(array(1,2,3,4,5,6),2,4),\n" + 
                "  subarray(array(1,2,3,4,5,6),0,2),\n" + 
                "  subarray(array(1,2,3,4,5,6),4,6),\n" + 
                "  subarray(array(1,2,3,4,5,6),4,7);\n" + 
                "\n" + 
                " [4,5]\n" + 
                " [3]\n" + 
                " []\n" + 
                " []\n" + 
                " [0,1]\n" + 
                " [0,1]\n" + 
                " [5,6]\n" + 
                " [5,6]\n" + 
                " [3,4]\n" + 
                " [1,2]\n" + 
                " [5,6]\n" + 
                " [5,6]")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class SubarrayUDF extends GenericUDF {

    private ListObjectInspector valuesOI;
    private PrimitiveObjectInspector fromIndexOI;
    @Nullable
    private PrimitiveObjectInspector toIndexOI;

    private final List<Object> result = new ArrayList<>();

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentLengthException(
                "Expected 2 or 3 arguments, but got " + argOIs.length);
        }

        this.valuesOI = HiveUtils.asListOI(argOIs[0]);
        this.fromIndexOI = HiveUtils.asIntegerOI(argOIs[1]);
        if (argOIs.length == 3) {
            this.toIndexOI = HiveUtils.asIntegerOI(argOIs[2]);
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
        result.clear();

        final int size = valuesOI.getListLength(arg0);

        Object arg1 = args[1].get();
        if (arg1 == null) {
            throw new UDFArgumentException("2nd argument MUST NOT be null");
        }
        int fromIndex = PrimitiveObjectInspectorUtils.getInt(arg1, fromIndexOI);
        if (fromIndex < 0) {
            fromIndex = 0;
        }

        int toIndex;
        if (args.length == 3) {
            Object arg2 = args[2].get();
            if (arg2 == null) {
                throw new UDFArgumentException("3rd argument MUST NOT be null");
            }
            toIndex = PrimitiveObjectInspectorUtils.getInt(arg2, toIndexOI);
            if (toIndex > size) {
                toIndex = size;
            }
        } else {
            toIndex = size;
        }

        for (int i = fromIndex; i < toIndex; i++) {
            Object e = valuesOI.getListElement(arg0, i);
            result.add(e);
        }

        return result;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "subarray(" + join(args, ',') + ")";
    }

}
