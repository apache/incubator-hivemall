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
package hivemall.tools.vector;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;

import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnegative;
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
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "vector_add",
        value = "_FUNC_(array<NUMBER> x, array<NUMBER> y) - Perform vector ADD operation.",
        extended = "SELECT vector_add(array(1.0,2.0,3.0), array(2, 3, 4));\n" + "[3.0,5.0,7.0]")
@UDFType(deterministic = true, stateful = false)
public final class VectorAddUDF extends GenericUDF {

    private ListObjectInspector xOI, yOI;
    private PrimitiveObjectInspector xElemOI, yElemOI;
    private boolean floatingPoints;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentLengthException("Expected 2 arguments, but got " + argOIs.length);
        }

        this.xOI = HiveUtils.asListOI(argOIs[0]);
        this.yOI = HiveUtils.asListOI(argOIs[1]);
        this.xElemOI = HiveUtils.asNumberOI(xOI.getListElementObjectInspector());
        this.yElemOI = HiveUtils.asNumberOI(yOI.getListElementObjectInspector());

        if (HiveUtils.isIntegerOI(xElemOI) && HiveUtils.isIntegerOI(yElemOI)) {
            this.floatingPoints = false;
            return ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaLongObjectInspector);
        } else {
            this.floatingPoints = true;
            return ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        }
    }

    @Nullable
    @Override
    public List<?> evaluate(@Nonnull DeferredObject[] args) throws HiveException {
        final Object arg0 = args[0].get();
        final Object arg1 = args[1].get();
        if (arg0 == null || arg1 == null) {
            return null;
        }

        final int xLen = xOI.getListLength(arg0);
        final int yLen = yOI.getListLength(arg1);
        if (xLen != yLen) {
            throw new HiveException(
                "vector lengths do not match. x=" + xOI.getList(arg0) + ", y=" + yOI.getList(arg1));
        }

        if (floatingPoints) {
            return evaluateDouble(arg0, arg1, xLen);
        } else {
            return evaluateLong(arg0, arg1, xLen);
        }
    }

    @Nonnull
    private List<Double> evaluateDouble(@Nonnull final Object vecX, @Nonnull final Object vecY,
            @Nonnegative final int size) {
        final Double[] arr = new Double[size];
        for (int i = 0; i < size; i++) {
            Object x = xOI.getListElement(vecX, i);
            Object y = yOI.getListElement(vecY, i);
            if (x == null || y == null) {
                continue;
            }
            double xd = PrimitiveObjectInspectorUtils.getDouble(x, xElemOI);
            double yd = PrimitiveObjectInspectorUtils.getDouble(y, yElemOI);
            double v = xd + yd;
            arr[i] = Double.valueOf(v);
        }
        return Arrays.asList(arr);
    }

    @Nonnull
    private List<Long> evaluateLong(@Nonnull final Object vecX, @Nonnull final Object vecY,
            @Nonnegative final int size) {
        final Long[] arr = new Long[size];
        for (int i = 0; i < size; i++) {
            Object x = xOI.getListElement(vecX, i);
            Object y = yOI.getListElement(vecY, i);
            if (x == null || y == null) {
                continue;
            }
            long xd = PrimitiveObjectInspectorUtils.getLong(x, xElemOI);
            long yd = PrimitiveObjectInspectorUtils.getLong(y, yElemOI);
            long v = xd + yd;
            arr[i] = Long.valueOf(v);
        }
        return Arrays.asList(arr);
    }

    @Override
    public String getDisplayString(String[] args) {
        return "vector_add(" + StringUtils.join(args, ',') + ")";
    }

}
