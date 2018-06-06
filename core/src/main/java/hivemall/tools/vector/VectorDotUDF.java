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

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "vector_dot",
        value = "_FUNC_(array<NUMBER> x, array<NUMBER> y) - Performs vector dot product.",
        extended = "SELECT vector_dot(array(1.0,2.0,3.0),array(2.0,3.0,4.0));\n20\n\n"
                + "SELECT vector_dot(array(1.0,2.0,3.0),2);\n[2.0,4.0,6.0]")
@UDFType(deterministic = true, stateful = false)
public final class VectorDotUDF extends GenericUDF {

    private Evaluator evaluator;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentLengthException("Expected 2 arguments, but got " + argOIs.length);
        }

        ObjectInspector argOI0 = argOIs[0];
        if (!HiveUtils.isNumberListOI(argOI0)) {
            throw new UDFArgumentException(
                "Expected array<number> for the first argument: " + argOI0.getTypeName());
        }
        ListObjectInspector xListOI = HiveUtils.asListOI(argOI0);

        ObjectInspector argOI1 = argOIs[1];
        if (HiveUtils.isNumberListOI(argOI1)) {
            this.evaluator = new Dot2DVectors(xListOI, HiveUtils.asListOI(argOI1));
            return PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
        } else if (HiveUtils.isNumberOI(argOI1)) {
            this.evaluator = new Multiply2D1D(xListOI, argOI1);
            return ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        } else {
            throw new UDFArgumentException(
                "Expected array<number> or number for the send argument: " + argOI1.getTypeName());
        }
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        final Object arg0 = args[0].get();
        final Object arg1 = args[1].get();
        if (arg0 == null || arg1 == null) {
            return null;
        }

        return evaluator.dot(arg0, arg1);
    }

    interface Evaluator extends Serializable {

        @Nonnull
        Object dot(@Nonnull Object x, @Nonnull Object y) throws HiveException;

    }

    static final class Multiply2D1D implements Evaluator {
        private static final long serialVersionUID = -9090211833041797311L;

        private final ListObjectInspector xListOI;
        private final PrimitiveObjectInspector xElemOI;
        private final PrimitiveObjectInspector yOI;

        Multiply2D1D(@Nonnull ListObjectInspector xListOI, @Nonnull ObjectInspector yOI)
                throws UDFArgumentTypeException {
            this.xListOI = xListOI;
            this.xElemOI = HiveUtils.asNumberOI(xListOI.getListElementObjectInspector());
            this.yOI = HiveUtils.asNumberOI(yOI);
        }

        @Override
        public List<Double> dot(@Nonnull Object x, @Nonnull Object y) throws HiveException {
            final double yd = PrimitiveObjectInspectorUtils.getDouble(y, yOI);

            final int xLen = xListOI.getListLength(x);
            final Double[] arr = new Double[xLen];
            for (int i = 0; i < xLen; i++) {
                Object xi = xListOI.getListElement(x, i);
                if (xi == null) {
                    continue;
                }
                double xd = PrimitiveObjectInspectorUtils.getDouble(xi, xElemOI);
                double v = xd * yd;
                arr[i] = Double.valueOf(v);
            }

            return Arrays.asList(arr);
        }

    }

    static final class Dot2DVectors implements Evaluator {
        private static final long serialVersionUID = -8783159823009951347L;

        private final ListObjectInspector xListOI, yListOI;
        private final PrimitiveObjectInspector xElemOI, yElemOI;

        Dot2DVectors(@Nonnull ListObjectInspector xListOI, @Nonnull ListObjectInspector yListOI)
                throws UDFArgumentTypeException {
            this.xListOI = xListOI;
            this.yListOI = yListOI;
            this.xElemOI = HiveUtils.asNumberOI(xListOI.getListElementObjectInspector());
            this.yElemOI = HiveUtils.asNumberOI(yListOI.getListElementObjectInspector());
        }

        @Override
        public Double dot(@Nonnull Object x, @Nonnull Object y) throws HiveException {
            final int xLen = xListOI.getListLength(x);
            final int yLen = yListOI.getListLength(y);
            if (xLen != yLen) {
                throw new HiveException("vector lengths do not match. x=" + xListOI.getList(x)
                        + ", y=" + yListOI.getList(y));
            }

            double result = 0.d;
            for (int i = 0; i < xLen; i++) {
                Object xi = xListOI.getListElement(x, i);
                Object yi = yListOI.getListElement(y, i);
                if (xi == null || yi == null) {
                    continue;
                }
                double xd = PrimitiveObjectInspectorUtils.getDouble(xi, xElemOI);
                double yd = PrimitiveObjectInspectorUtils.getDouble(yi, yElemOI);
                double v = xd * yd;
                result += v;
            }

            return Double.valueOf(result);
        }

    }

    @Override
    public String getDisplayString(String[] args) {
        return "vector_dot(" + StringUtils.join(args, ',') + ")";
    }

}
