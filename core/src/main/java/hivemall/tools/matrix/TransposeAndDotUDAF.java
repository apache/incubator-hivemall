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
package hivemall.tools.matrix;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.SizeOf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

// @formatter:off
@Description(name = "transpose_and_dot",
        value = "_FUNC_(array<number> X, array<number> Y)"
                + " - Returns dot(X.T, Y) as array<array<double>>, shape = (X.#cols, Y.#cols)",
        extended = "WITH input as (\n" + 
                "  select array(1.0, 2.0, 3.0, 4.0) as x, array(1, 2) as y\n" + 
                "  UNION ALL\n" + 
                "  select array(2.0, 3.0, 4.0, 5.0) as x, array(1, 2) as y\n" + 
                ")\n" + 
                "select\n" + 
                "  transpose_and_dot(x, y) as xy,\n" + 
                "  transpose_and_dot(y, x) as yx\n" + 
                "from \n" + 
                "  input;\n\n" + 
                "[[\"3.0\",\"6.0\"],[\"5.0\",\"10.0\"],[\"7.0\",\"14.0\"],[\"9.0\",\"18.0\"]]" + 
                "   [[\"3.0\",\"5.0\",\"7.0\",\"9.0\"],[\"6.0\",\"10.0\",\"14.0\",\"18.0\"]]\n")
// @formatter:on
public final class TransposeAndDotUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        ObjectInspector[] OIs = info.getParameterObjectInspectors();

        if (OIs.length != 2) {
            throw new UDFArgumentLengthException("Specify two arguments.");
        }

        if (!HiveUtils.isNumberListOI(OIs[0])) {
            throw new UDFArgumentTypeException(0,
                "Only array<number> type argument is acceptable but " + OIs[0].getTypeName()
                        + " was passed as `matrix0_row`");
        }

        if (!HiveUtils.isNumberListOI(OIs[1])) {
            throw new UDFArgumentTypeException(1,
                "Only array<number> type argument is acceptable but " + OIs[1].getTypeName()
                        + " was passed as `matrix1_row`");
        }

        return new TransposeAndDotUDAFEvaluator();
    }

    static final class TransposeAndDotUDAFEvaluator extends GenericUDAFEvaluator {
        // PARTIAL1 and COMPLETE
        private ListObjectInspector xRowOI;
        private PrimitiveObjectInspector xElemOI;
        private ListObjectInspector yRowOI;
        private PrimitiveObjectInspector yElemOI;

        // PARTIAL2 and FINAL
        private ListObjectInspector aggMatrixOI;
        private ListObjectInspector aggMatrixRowOI;
        private DoubleObjectInspector aggMatrixElemOI;

        private double[] xRow;
        private double[] yRow;

        @AggregationType(estimable = true)
        static class TransposeAndDotAggregationBuffer extends AbstractAggregationBuffer {
            double[][] aggMatrix;

            @Override
            public int estimate() {
                return aggMatrix != null ? aggMatrix.length * aggMatrix[0].length * SizeOf.DOUBLE
                        : 0;
            }

            public void init(int n, int m) {
                this.aggMatrix = new double[n][m];
            }

            public void reset() {
                if (aggMatrix != null) {
                    for (double[] row : aggMatrix) {
                        Arrays.fill(row, 0.d);
                    }
                }
            }
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] OIs) throws HiveException {
            super.init(mode, OIs);

            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {
                this.xRowOI = HiveUtils.asListOI(OIs[0]);
                this.xElemOI =
                        HiveUtils.asDoubleCompatibleOI(xRowOI.getListElementObjectInspector());
                this.yRowOI = HiveUtils.asListOI(OIs[1]);
                this.yElemOI =
                        HiveUtils.asDoubleCompatibleOI(yRowOI.getListElementObjectInspector());
            } else {
                this.aggMatrixOI = HiveUtils.asListOI(OIs[0]);
                this.aggMatrixRowOI =
                        HiveUtils.asListOI(aggMatrixOI.getListElementObjectInspector());
                this.aggMatrixElemOI =
                        HiveUtils.asDoubleOI(aggMatrixRowOI.getListElementObjectInspector());
            }

            return ObjectInspectorFactory.getStandardListObjectInspector(
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        }

        @Override
        public AbstractAggregationBuffer getNewAggregationBuffer() throws HiveException {
            TransposeAndDotAggregationBuffer myAgg = new TransposeAndDotAggregationBuffer();
            reset(myAgg);
            return myAgg;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            TransposeAndDotAggregationBuffer myAgg = (TransposeAndDotAggregationBuffer) agg;
            myAgg.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            final Object matrix0RowObj = parameters[0];
            final Object matrix1RowObj = parameters[1];

            // need to care about NULL since NULL is passed when aggregating with zero row.
            if (matrix0RowObj == null || matrix1RowObj == null) {
                return;
            }

            final TransposeAndDotAggregationBuffer myAgg = (TransposeAndDotAggregationBuffer) agg;

            if (xRow == null) {
                xRow = new double[xRowOI.getListLength(matrix0RowObj)];
            }
            if (yRow == null) {
                yRow = new double[yRowOI.getListLength(matrix1RowObj)];
            }

            HiveUtils.toDoubleArray(matrix0RowObj, xRowOI, xElemOI, xRow, false);
            HiveUtils.toDoubleArray(matrix1RowObj, yRowOI, yElemOI, yRow, false);

            if (myAgg.aggMatrix == null) {
                myAgg.init(xRow.length, yRow.length);
            }

            for (int i = 0; i < xRow.length; i++) {
                for (int j = 0; j < yRow.length; j++) {
                    myAgg.aggMatrix[i][j] += xRow[i] * yRow[j];
                }
            }
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object other)
                throws HiveException {
            if (other == null) {
                return;
            }

            final TransposeAndDotAggregationBuffer myAgg = (TransposeAndDotAggregationBuffer) agg;

            final List<?> matrix = aggMatrixOI.getList(other);
            final int n = matrix.size();
            final double[] row = new double[aggMatrixRowOI.getListLength(matrix.get(0))];
            for (int i = 0; i < n; i++) {
                HiveUtils.toDoubleArray(matrix.get(i), aggMatrixRowOI, aggMatrixElemOI, row, false);

                if (myAgg.aggMatrix == null) {
                    myAgg.init(n, row.length);
                }

                for (int j = 0; j < row.length; j++) {
                    myAgg.aggMatrix[i][j] += row[j];
                }
            }
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            return terminate(agg);
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            final TransposeAndDotAggregationBuffer myAgg = (TransposeAndDotAggregationBuffer) agg;

            if (myAgg.aggMatrix == null) {
                return null;
            }

            final List<List<DoubleWritable>> result = new ArrayList<List<DoubleWritable>>();
            for (double[] row : myAgg.aggMatrix) {
                result.add(WritableUtils.toWritableList(row));
            }
            return result;
        }
    }
}
