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
package hivemall.tools.aggr;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.util.JavaDataModel;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;

//@formatter:off
@Description(name = "max_by",
        value = "_FUNC_(x, y) - Returns the value of x associated with the maximum value of y over all input values.",
        extended = "-- see https://issues.apache.org/jira/browse/HIVE-17406 \n"
                + "WITH data as (\n" + 
                "  select 'jake' as name, 18 as age\n" + 
                "  union all\n" + 
                "  select 'tom' as name, 64 as age\n" + 
                "  union all\n" + 
                "  select 'lisa' as name, 32 as age\n" + 
                ")\n" + 
                "select\n" + 
                "  max_by(name, age) as name\n" + 
                "from\n" + 
                "  data;\n" + 
                "tom")
//@formatter:on
public final class MaxByUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] argTypes)
            throws SemanticException {
        if (argTypes.length != 2) {
            throw new UDFArgumentLengthException(
                "Exactly two arguments are expected: " + argTypes.length);
        }
        ObjectInspector yOI = TypeInfoUtils.getStandardJavaObjectInspectorFromTypeInfo(argTypes[1]);
        if (!ObjectInspectorUtils.compareSupported(yOI)) {
            throw new UDFArgumentTypeException(1,
                "Cannot support comparison of map<> type or complex type containing map<>.");
        }
        return new Evaluator();
    }

    @UDFType(distinctLike = true)
    public static class Evaluator extends GenericUDAFEvaluator {

        private transient ObjectInspector xInputOI, yInputOI;
        private transient ObjectInspector xOutputOI, yOutputOI;

        @Nullable
        private transient StructField xField, yField;
        @Nullable
        private transient StructObjectInspector partialInputOI;

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] argOIs) throws HiveException {
            super.init(mode, argOIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.xInputOI = argOIs[0];
                this.yInputOI = argOIs[1];
                if (!ObjectInspectorUtils.compareSupported(yInputOI)) {
                    throw new UDFArgumentTypeException(1,
                        "Cannot support comparison of map<> type or complex type containing map<>.");
                }
            } else {// from partial aggregation
                this.partialInputOI = (StructObjectInspector) argOIs[0];
                this.xField = partialInputOI.getStructFieldRef("x");
                this.xInputOI = xField.getFieldObjectInspector();
                this.yField = partialInputOI.getStructFieldRef("y");
                this.yInputOI = yField.getFieldObjectInspector();
            }
            this.xOutputOI = ObjectInspectorUtils.getStandardObjectInspector(xInputOI,
                ObjectInspectorCopyOption.JAVA);
            this.yOutputOI = ObjectInspectorUtils.getStandardObjectInspector(yInputOI,
                ObjectInspectorCopyOption.JAVA);

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                List<String> fieldNames = new ArrayList<>(2);
                List<ObjectInspector> fieldOIs = new ArrayList<>(2);
                fieldNames.add("x");
                fieldOIs.add(xOutputOI);
                fieldNames.add("y");
                fieldOIs.add(yOutputOI);
                return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames,
                    fieldOIs);
            } else {// terminate
                // Copy to Java object because that saves object creation time.
                outputOI = ObjectInspectorUtils.getStandardObjectInspector(xInputOI,
                    ObjectInspectorCopyOption.JAVA);
            }
            return outputOI;
        }

        /** class for storing the current max value */
        @AggregationType(estimable = true)
        static class MaxAgg extends AbstractAggregationBuffer {
            Object x, y;

            @Override
            public int estimate() {
                return JavaDataModel.PRIMITIVES2 * 2; // rough estimate
            }

            void merge(final Object newX, final Object newY,
                    @Nonnull final ObjectInspector xInputOI,
                    @Nonnull final ObjectInspector yInputOI,
                    @Nonnull final ObjectInspector yOutputOI) {
                final int cmp = ObjectInspectorUtils.compare(y, yOutputOI, newY, yInputOI);
                if (x == null || cmp < 0) { // found greater y
                    this.x = ObjectInspectorUtils.copyToStandardObject(newX, xInputOI,
                        ObjectInspectorCopyOption.JAVA);
                    this.y = ObjectInspectorUtils.copyToStandardObject(newY, yInputOI,
                        ObjectInspectorCopyOption.JAVA);
                }
            }
        }

        @Override
        public MaxAgg getNewAggregationBuffer() throws HiveException {
            return new MaxAgg();
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MaxAgg myagg = (MaxAgg) agg;
            myagg.x = null;
            myagg.y = null;
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            assert (parameters.length == 2);
            MaxAgg myagg = (MaxAgg) agg;
            Object x = parameters[0];
            Object y = parameters[1];

            myagg.merge(x, y, xInputOI, yInputOI, yOutputOI);
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MaxAgg myagg = (MaxAgg) agg;
            Object[] partial = new Object[2];
            partial[0] = myagg.x;
            partial[1] = myagg.y;
            return partial;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object partial)
                throws HiveException {
            if (partial == null) {
                return;
            }

            MaxAgg myagg = (MaxAgg) agg;
            Object x = partialInputOI.getStructFieldData(partial, xField);
            Object y = partialInputOI.getStructFieldData(partial, yField);

            myagg.merge(x, y, xInputOI, yInputOI, yOutputOI);
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            MaxAgg myagg = (MaxAgg) agg;
            return myagg.x;
        }

    }

}
